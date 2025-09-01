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

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="not used in jittor, set via jt.flags.use_cuda")
opt = parser.parse_args()

# Jittor handles device placement automatically. Setting flags is the preferred way.
# os.environ["CUDA_VISIBLE_DEVICES"] is not necessary as Jittor manages this internally.
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.n_gpu
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
        batch_imgs, batch_boxes, batch_classes = data
        
        lr = lr_func(GLOBAL_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        start_time = time.time()
        
        # Jittor's `execute()` method is called implicitly when the module is called.
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        
        # Forward and backward pass
        optimizer.zero_grad()
        loss.mean().backward()
        
        # Gradient clipping
        jt.grad.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        
        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" %
              (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean().item(), losses[1].mean().item(), losses[2].mean().item(), cost_time, lr, loss.mean().item()))

        GLOBAL_STEPS += 1
    
    jt.save(model.state_dict(), "./checkpoint/model_{}.pth".format(epoch + 1))
