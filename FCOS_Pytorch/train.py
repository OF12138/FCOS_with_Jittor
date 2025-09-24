from model.FCOS import FCOSDetector
import torch
from dataset.dataset import COCODataset
import math,time
from dataset.augmentation import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser() #allow the command line arguments
#add command line arguments
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args() #store the arguments in opt

#set the environment
os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu

#set the dataset
transform = Transforms()
train_dataset=COCODataset("./dataset/tiny_coco/images/train2017",
                          './dataset/tiny_coco/annotations/instances_train2017.json',
                          transform=transform,
                          )
#set the dataloader
train_loader=torch.utils.data.DataLoader(train_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         collate_fn=train_dataset.collate_fn, #to tensor
                                         num_workers=opt.n_cpu,
                                         )

writer=SummaryWriter('logs/FCOS_experiment_pytorch_final')
model=FCOSDetector(mode="training").cuda() #instance of model
BATCH_SIZE=opt.batch_size
EPOCHS=opt.epochs

steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMUP_STEPS=500
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS=0
LR_INIT=0.01
optimizer = torch.optim.SGD(model.parameters(),
                            lr =LR_INIT,
                            momentum=0.9,
                            weight_decay=0.0001, #to prevent overfitting
                            )

#set the learning rate scheduler
lr_schedule = [120000, 160000]
#define the learning rate scheduler
def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)

#start to train
model.train()

for epoch in range(EPOCHS):
    for epoch_step,data in enumerate(train_loader):
        #get the data
        batch_imgs,batch_boxes,batch_classes=data
        batch_imgs=batch_imgs.cuda()
        batch_boxes=batch_boxes.cuda()
        batch_classes=batch_classes.cuda()

        lr = lr_func(GLOBAL_STEPS) #compute the learing rate 
        #update the lr of optimizer 
        for param in optimizer.param_groups:
            param['lr']=lr
        
        start_time=time.time() 

        #forward and backpropagation
        optimizer.zero_grad()
        losses=model([batch_imgs,batch_boxes,batch_classes])
        loss=losses[-1]

        writer.add_scalar('Loss/Total', loss.mean().item(), GLOBAL_STEPS)
        writer.add_scalar('Loss/Classification', losses[0].mean().item(), GLOBAL_STEPS)
        writer.add_scalar('Loss/Centerness', losses[1].mean().item(), GLOBAL_STEPS)
        writer.add_scalar('Loss/Regression', losses[2].mean().item(), GLOBAL_STEPS)
        writer.add_scalar('LearningRate', lr, GLOBAL_STEPS)


        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3)  #prevents the "exploding gradient", the 3 is the threshold
        optimizer.step()

        end_time=time.time()
        cost_time=int((end_time-start_time)*1000) #convert second to ms

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f"%\
            (GLOBAL_STEPS,epoch+1,epoch_step+1,steps_per_epoch,losses[0].mean(),losses[1].mean(),losses[2].mean(),cost_time,lr, loss.mean()))


        GLOBAL_STEPS+=1

    writer.close()
    
    torch.save(model.state_dict(),"./checkpoint/model_{}.pth".format(epoch+1))
    