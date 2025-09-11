from model.FCOS import FCOSDetector
import jittor as jt
from jittor.dataset import DataLoader
import math, time
from dataset.dataset import COCODataset
from dataset.augmentation import Transforms
import os
import numpy as np
import random
import argparse
from jittor import nn
#from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="not used in jittor, set via jt.flags.use_cuda")
opt = parser.parse_args()

# Jittor handles device placement automatically. Setting flags is the preferred way.
# os.environ["CUDA_VISIBLE_DEVICES"] is not necessary as Jittor manages this internally.
#if 'CUDA_VISIBLE_DEVICES' in os.environ:
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.n_gpu
jt.flags.use_cuda = 1

# Setting seeds for reproducibility in Jittor
jt.set_global_seed(0)
np.random.seed(0)
random.seed(0)

# set the dataset
transform = Transforms()
train_dataset = COCODataset("./dataset/tiny_coco/images/train2017",
                            './dataset/tiny_coco/annotations/instances_train2017.json',
                            transform=transform,
                        )
# set the dataloader
train_loader = DataLoader(train_dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=opt.n_cpu,
                        )

#writer=SummaryWriter('logs/FCOS_experiment_1')
model = FCOSDetector(mode="training")
# Jittor's DataParallel is not explicitly needed as it handles this automatically.
# model = jt.nn.DataParallel(model) 

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs

steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMUP_STEPS = 500
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS = 0
LR_INIT = 0.01

optimizer = jt.optim.SGD(model.parameters(),
                        lr=LR_INIT,
                        momentum=0.9,
                        weight_decay=0.0001,
                        )

# set the learning rate scheduler
lr_schedule = [120000, 160000]

def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr * warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)

# start to train
model.train()

for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):
        # Data is already on the correct device with Jittor, no need for .cuda()
        batch_imgs, batch_boxes, batch_classes ,batch_scales = data
        #print("batch_imgs.shape before model:", batch_imgs.shape)
        #print(batch_imgs.dtype)


        lr = lr_func(GLOBAL_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        start_time = time.time()

        # temporarily in training script before model call:
        #print("cls_logits levels:", [p.shape for p in cls_logits])
        #print("batch_boxes shape:", batch_boxes.shape, "batch_classes shape:", batch_classes.shape)
        #print("cls_logits levels shapes:", [p.shape for p in batch_imgs[0]])
        #print("batch_boxes:", batch_boxes.shape, "batch_classes:", batch_classes.shape)


        
        # Jittor's `execute()` method is called implicitly when the module is called.
        losses = model([batch_imgs, batch_boxes, batch_classes, batch_scales])
        loss = losses[-1]

        # Extract individual losses (ensure these indices match your model's output)
        cls_loss = losses[0].mean()
        cnt_loss = losses[1].mean()
        reg_loss = losses[2].mean()
        total_loss = losses[-1].mean()  # Total loss

        #writer.add_scalar('Loss/Total', total_loss, GLOBAL_STEPS)
        #writer.add_scalar('Loss/Classification', cls_loss, GLOBAL_STEPS)
        #writer.add_scalar('Loss/Regression', reg_loss, GLOBAL_STEPS)
        #writer.add_scalar('Loss/Centerness', cnt_loss, GLOBAL_STEPS)
        #writer.add_scalar('LearningRate', lr, GLOBAL_STEPS)
        
        # Check if current step has meaningful loss
        if cls_loss.item() > 0 or cnt_loss.item() > 0 or reg_loss.item() > 0:
            epoch_has_meaningful_loss = True
        
        # Forward and backward pass
        #optimizer.zero_grad()
        #optimizer.backward(loss.mean())
        
        # Gradient clipping
        #jt.grad.clip_grad_norm_(model.parameters(), 3)
        #optimizer.step()

        #jt.grad.clip_grad_norm_(model.parameters(),3)
        #nn.clip_grad_norm_(model.parameters(),3)

        optimizer.step(loss.mean())


        
        
        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" %
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean().item(), losses[1].mean().item(), losses[2].mean().item(), cost_time, lr, loss.mean().item()))

        GLOBAL_STEPS += 1
    
    # Save model with corrected condition
    # Check if training made progress (any loss component is positive)
    if GLOBAL_STEPS > 0 and epoch_has_meaningful_loss:
        try:
            # Create checkpoint directory if it doesn't exist
            os.makedirs("./checkpoint", exist_ok=True)
            jt.save(model.state_dict(), f"./checkpoint/model_{epoch + 1}.pth")
            print(f"Model saved successfully at epoch {epoch + 1}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    else:
        print(f"Skipping model save at epoch {epoch + 1} - no meaningful training progress")
