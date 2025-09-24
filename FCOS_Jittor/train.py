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
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="not used in jittor, set via jt.flags.use_cuda")
opt = parser.parse_args()

jt.flags.use_cuda = 1

# Setting seeds for reproducibility in Jittor
jt.set_global_seed(0)
np.random.seed(0)
random.seed(0)

# set the dataset
transform = Transforms()
train_dataset = COCODataset("./dataset/tiny_coco/images/train2017",
                            './dataset/tiny_coco/annotations/instances_train2017.json',
                            #"/home/openfar/Dataset/COCO/train2017/train2017",
                            #"/home/openfar/Dataset/COCO/annotations/instances_train2017.json",
                            transform=transform,
                        )
# set the dataloader
train_loader = DataLoader(train_dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        )

writer=SummaryWriter('logs/FCOS_experiment_jittor_easy_2')
model = FCOSDetector(mode="training")

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
# train.py (New Version)
""" optimizer = jt.optim.AdamW(model.parameters(),
                           lr=LR_INIT,
                           weight_decay=0.0001 
                          ) """

# set the learning rate scheduler
lr_schedule = [120000,160000]

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

        # --- START: ADD THIS DEBUG BLOCK ---
        if GLOBAL_STEPS < 2: # Print for the first 2 steps
            print("\n" + "="*40)
            print(f"DEBUG INFO FOR: global_step={GLOBAL_STEPS}")
            print("="*40)
            print(f"[TRAIN] Image batch shape: {batch_imgs.shape}, Scales: {batch_scales.numpy().flatten().tolist()}")
            num_boxes_to_print = min(2, batch_boxes.shape[1])
            if batch_boxes.numel() > 0 and batch_boxes[0,0,0].item() != -1:
                print(f"[TRAIN] Sample GT Boxes (first {num_boxes_to_print}):\n{batch_boxes[0, :num_boxes_to_print, :].numpy()}")
                print(f"[TRAIN] Sample GT Classes (first {num_boxes_to_print}): {batch_classes[0, :num_boxes_to_print].numpy()}")
            else:
                print("[TRAIN] No valid GT Boxes in this batch.")
        # --- END: ADD THIS DEBUG BLOCK ---


        lr = lr_func(GLOBAL_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        start_time = time.time()

        # Jittor's `execute()` method is called implicitly when the module is called.
        losses = model([batch_imgs, batch_boxes, batch_classes, batch_scales])
        loss = losses[-1]

        #the visualization
        # --- 2. ADD THIS BLOCK inside your training loop ---
        # After calculating the losses, log them to TensorBoard
        writer.add_scalar('Loss/Total', loss.mean().item(), GLOBAL_STEPS)
        writer.add_scalar('Loss/Classification', losses[0].mean().item(), GLOBAL_STEPS)
        writer.add_scalar('Loss/Centerness', losses[1].mean().item(), GLOBAL_STEPS)
        writer.add_scalar('Loss/Regression', losses[2].mean().item(), GLOBAL_STEPS)
        writer.add_scalar('LearningRate', lr, GLOBAL_STEPS)
        # --- END OF BLOCK ---

        # Extract individual losses (ensure these indices match your model's output)
        cls_loss = losses[0].mean()
        cnt_loss = losses[1].mean()
        reg_loss = losses[2].mean()
        total_loss = losses[-1].mean()  # Total loss

        # Check if current step has meaningful loss
        if cls_loss.item() > 0 or cnt_loss.item() > 0 or reg_loss.item() > 0:
            epoch_has_meaningful_loss = True

        #optimizer.step(loss.mean())
        #optimizer.zero_grad()

        optimizer.backward(loss.mean())
        optimizer.clip_grad_norm(max_norm=3.0) # Clip the gradients
        optimizer.step() # Apply the clipped gradients
        

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" %
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean().item(), losses[1].mean().item(), losses[2].mean().item(), cost_time, lr, loss.mean().item()))

        GLOBAL_STEPS += 1

    writer.close()
    
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
