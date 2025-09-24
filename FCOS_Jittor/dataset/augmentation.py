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
    # Rotates the image randomly between -10 and 10 degrees.
    d = random.uniform(-10, 10)
    w, h = img.size

    rx0, ry0 = w / 2.0, h / 2.0
    # Use a fill color to avoid black borders, matching PyTorch's potential behavior
    img = img.rotate(d, fillcolor=(128,128,128)) 

    a = -d / 180.0 * math.pi

    new_boxes = np.zeros_like(boxes)
    # Assuming input boxes are [xmin, ymin, xmax, ymax]
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i, :]

        points = np.array([
            [xmin, ymin], [xmin, ymax],
            [xmax, ymin], [xmax, ymax]
        ])

        # Rotate points
        new_points = np.zeros_like(points)
        for j, (x, y) in enumerate(points):
            new_points[j, 0] = (x - rx0) * math.cos(a) - (y - ry0) * math.sin(a) + rx0
            new_points[j, 1] = (x - rx0) * math.sin(a) + (y - ry0) * math.cos(a) + ry0

        # Get new bounding box from the min/max of the rotated points
        new_xmin = np.min(new_points[:, 0])
        new_ymin = np.min(new_points[:, 1])
        new_xmax = np.max(new_points[:, 0])
        new_ymax = np.max(new_points[:, 1])

        new_boxes[i] = [new_xmin, new_ymin, new_xmax, new_ymax]

    # Clamp boxes to stay within the image dimensions
    new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, w - 1)
    new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, h - 1)

    # Return a PIL Image and a NumPy array, just like the other augmentations
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




#---test---





if __name__ == '__main__':
    try:
        import torch
        import torchvision
    except ImportError:
        print("PyTorch or Torchvision not found. Skipping comparison test.")
        exit()

    # --- START: Self-Contained PyTorch Implementation ---
    def colorJitter_pt(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        img = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)
        return img, boxes

    def random_rotation_pt(img, boxes, degree=10):
        d = random.uniform(-degree, degree)
        w, h = img.size
        rx0, ry0 = w / 2.0, h / 2.0
        img = img.rotate(d, fillcolor=(128,128,128))
        a = -d / 180.0 * math.pi
        boxes = torch.from_numpy(boxes)
        new_boxes = torch.zeros_like(boxes)
        for i in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[i, :]
            points = torch.tensor([
                [xmin, ymin], [xmin, ymax],
                [xmax, ymin], [xmax, ymax]
            ], dtype=torch.float32)
            new_points = torch.zeros_like(points)
            for j, (px, py) in enumerate(points):
                new_points[j, 0] = (px - rx0) * math.cos(a) - (py - ry0) * math.sin(a) + rx0
                new_points[j, 1] = (px - rx0) * math.sin(a) + (py - ry0) * math.cos(a) + ry0
            
            new_xmin, new_ymin = torch.min(new_points, dim=0)[0]
            new_xmax, new_ymax = torch.max(new_points, dim=0)[0]
            new_boxes[i] = torch.tensor([new_xmin, new_ymin, new_xmax, new_ymax])

        new_boxes[:, [0, 2]].clamp_(min=0, max=w - 1)
        new_boxes[:, [1, 3]].clamp_(min=0, max=h - 1)
        return img, new_boxes.numpy()

    def _box_inter_pt(box1, box2):
        tl = torch.max(box1[:,None,:2], box2[:,:2])
        br = torch.min(box1[:,None,2:], box2[:,2:])
        hw = (br-tl).clamp(min=0)
        return hw[:,:,0] * hw[:,:,1]

    def random_crop_resize_pt(img, boxes, crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, attempt_max=10):
        success = False
        boxes_pt = torch.from_numpy(boxes)
        for _ in range(attempt_max):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(crop_scale_min, 1.0) * area
            aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
            w = int(round(math.sqrt(target_area * aspect_ratio_)))
            h = int(round(math.sqrt(target_area / aspect_ratio_)))
            if random.random() < 0.5: w, h = h, w
            if w <= img.size[0] and h <= img.size[1]:
                x = random.randint(0, img.size[0] - w)
                y = random.randint(0, img.size[1] - h)
                crop_box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
                inter = _box_inter_pt(crop_box, boxes_pt)
                box_area = (boxes_pt[:, 2]-boxes_pt[:, 0])*(boxes_pt[:, 3]-boxes_pt[:, 1])
                if box_area.numel() > 0:
                    mask = inter > 0.0001
                    inter = inter[mask]
                    box_area = box_area[mask.squeeze(0)]
                    box_remain = inter.squeeze(0) / box_area
                    if box_remain.numel() > 0 and torch.all(box_remain > remain_min):
                        success = True; break
                else:
                    success = True; break
        if success:
            img = img.crop((x, y, x+w, y+h))
            boxes -= np.array([x,y,x,y], dtype=np.float32)
            boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, h-1)
            boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, w-1)
        return img, boxes
    # --- END: Self-Contained PyTorch Implementation ---

    # --- Create Mock Data ---
    print("--- Creating Mock Data ---")
    mock_image = Image.new('RGB', (800, 600), color = 'red')
    mock_boxes = np.array([
        [100, 150, 300, 400],
        [500, 50, 600, 200]
    ], dtype=np.float32)
    print(f"Initial Image Size: {mock_image.size}")
    print(f"Initial Boxes:\n{mock_boxes}")

    def compare_augmentations(aug_name, jt_func, pt_func, seed):
        print(f"\n--- Testing Augmentation: {aug_name} ---")
        random.seed(seed); np.random.seed(seed); jt.set_global_seed(seed)
        img_jt, boxes_jt = jt_func(mock_image.copy(), mock_boxes.copy())
        
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        img_pt, boxes_pt = pt_func(mock_image.copy(), mock_boxes.copy())

        print(f"Jittor output image size: {img_jt.size}")
        print(f"PyTorch output image size: {img_pt.size}")
        print(f"Jittor output boxes:\n{boxes_jt}")
        print(f"PyTorch output boxes:\n{boxes_pt}")

        img_match = (img_jt.size == img_pt.size and img_jt.mode == img_pt.mode)
        boxes_match = np.allclose(boxes_jt, boxes_pt, atol=1e-5)

        if img_match and boxes_match:
            print("✅ Outputs MATCH")
            return True
        else:
            print("❌ Outputs DO NOT MATCH")
            if not img_match: print("   - Image size or mode differs.")
            if not boxes_match: print(f"   - Max box difference: {np.abs(boxes_jt - boxes_pt).max()}")
            return False

    # --- Run Comparisons ---
    seed = 42
    results = [
        compare_augmentations("Color Jitter", colorJitter, colorJitter_pt, seed),
        compare_augmentations("Random Rotation", random_rotation, random_rotation_pt, seed),
        compare_augmentations("Random Crop & Resize", random_crop_resize, random_crop_resize_pt, seed)
    ]
    
    print("\n--- Final Result ---")
    if all(results):
        print("🎉🎉🎉 All augmentations are equivalent.")
    else:
        print("🔥🔥🔥 One or more augmentations have different behavior.")



