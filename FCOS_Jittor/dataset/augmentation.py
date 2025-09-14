import jittor as jt
import jittor.transform as transforms
import math
from PIL import Image
import random
import numpy as np
import cv2

class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        if random.random() < 0.3:
            img, boxes = colorJitter(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_rotation(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_crop_resize(img, boxes)
        return img, boxes

def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    img = transforms.ColorJitter(brightness=brightness,
                                 contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, boxes

def random_rotation(img, boxes):
    # Convert image to numpy array if it's not already.
    # The 'shape' attribute does not exist on a PIL 'Image' object.
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Rotates the image randomly between -10 and 10 degrees.
    angle = random.randint(-10, 10)
    h, w, c = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))

    # Convert rotated image to Jittor tensor, normalize, and transpose dimensions.
    img = jt.array(img).float32() / 255.0
    img = jt.transpose(img, [2, 0, 1])

    # Get rotated points
    x = boxes[:, [0, 2]]
    y = boxes[:, [1, 3]]
    tp = jt.array([[x[0], y[0]], [x[1], y[0]], [x[0], y[1]], [x[1], y[1]]])
    tp = jt.reshape(tp, (-1, 2)).transpose(1, 0)
    
    # Pad points with 1 for matrix multiplication
    tp = jt.concat((tp, jt.ones((1, tp.shape[1]))), dim=0)

    # Convert rotation matrix to Jittor Var for multiplication
    M = jt.array(M)
    new_points = M @ tp

    # Calculate new bounding box
    # The original code caused an error due to incorrect unpacking.
    # We now find the maximum x and y coordinates separately.
    ymax = jt.max(new_points[0]).item()
    xmax = jt.max(new_points[1]).item()
    ymin = jt.min(new_points[0]).item()
    xmin = jt.min(new_points[1]).item()

    # Reformat the bounding boxes
    new_boxes = jt.stack([ymin, xmin, ymax, xmax]).transpose(0, 1)
    return img, new_boxes



def _box_inter(box1, box2):
    tl = jt.maximum(box1[:, None, :2], box2[:, :2])  # [n,m,2]
    br = jt.minimum(box1[:, None, 2:], box2[:, 2:])  # [n,m,2]
    hw = (br - tl).clamp(min_v=0)  # [n,m,2]
    inter = hw[:, :, 0] * hw[:, :, 1]  # [n,m]
    return inter

def random_crop_resize(img, boxes, crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, attempt_max=10):
    success = False
    boxes = jt.array(boxes)
    for attempt in range(attempt_max):
        # choose crop size
        area = img.size[0] * img.size[1]
        target_area = random.uniform(crop_scale_min, 1.0) * area
        aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
        w = int(round(math.sqrt(target_area * aspect_ratio_)))
        h = int(round(math.sqrt(target_area / aspect_ratio_)))
        if random.random() < 0.5:
            w, h = h, w
        # if size is right then random crop
        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            # check
            crop_box = jt.float32([[x, y, x + w, y + h]])
            inter = _box_inter(crop_box, boxes)  # [1,N] N can be zero
            box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
            
            mask = inter > 0.0001  # [1,N] N can be zero
            inter = inter[mask]  # [1,S] S can be zero
            box_area = box_area[mask.reshape(-1)]  # [S]
            box_remain = inter.reshape(-1) / box_area  # [S]
            
            if box_remain.shape[0] != 0:
                if bool(jt.min(box_remain > remain_min)):
                    success = True
                    break
            else:
                success = True
                break
    
    if success:
        img = img.crop((x, y, x + w, y + h))
        boxes -= jt.float32([x, y, x, y])
        
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min_v=0, max_v=h - 1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min_v=0, max_v=w - 1)
        
    boxes = boxes.numpy()
    return img, boxes