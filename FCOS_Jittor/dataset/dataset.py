import jittor as jt
from jittor.dataset import Dataset
import jittor.transform as transforms
from pycocotools.coco import COCO
import numpy as np
import cv2
from PIL import Image
import random
import os
from model.config import DefaultConfig


# -----------------------------
# Utils
# -----------------------------
def flip(img, boxes):
    """Flip an image and its bounding boxes horizontally."""
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0], boxes[:, 2] = xmin, xmax
    return img, boxes


def preprocess_img_boxes(image, boxes, input_size):
    """Resize image + boxes with padding, keep aspect ratio. Output CHW np.array."""
    if isinstance(input_size, (list, tuple)):
        min_side, max_side = input_size
    else:
        min_side, max_side = input_size, input_size

    h, w, c = image.shape
    if h < 10 or w < 10:  # edge case: very small image
        new_image = np.full((max_side, max_side, 3), 128, dtype=np.uint8)
        return np.transpose(new_image, (2, 0, 1)), boxes

    # Calculate scale correctly
    scale = min(min_side / min(h, w), max_side / max(h, w))
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))
    
    # Create padded image
    new_image = np.full((max_side, max_side, 3), 128, dtype=np.uint8)
    new_image[:nh, :nw, :] = image_resized

    # ... (existing code)
    # Add debug print to check scaling
    print(f"Original image size: ({w}, {h}), scale: {scale:.4f}")
    print(f"Resized image size: ({nw}, {nh}), padded size: ({max_side}, {max_side})")
    if boxes.shape[0] > 0:
        print(f"Scaled boxes sample: {boxes[:2].tolist()}")

    # Adjust boxes with the same scale
    if boxes.shape[0] > 0:
        boxes[:, :4] *= scale
    return np.transpose(new_image, (2, 0, 1)), boxes ,scale


# -----------------------------
# Dataset
# -----------------------------
class COCODataset(Dataset):
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, imgs_path, anno_path, resize_size=[800, 1333], is_train=True, transform=None):
        super().__init__()
        self.imgs_path = imgs_path
        self.coco = COCO(anno_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids

        # Map coco category_id → contiguous id (1..num_classes-1). 0 = background
        # replace the old mapping block
        cat_ids = sorted(self.coco.getCatIds())            # make order deterministic
        self.category2id = {v: i+1 for i, v in enumerate(cat_ids)}   # 0..79
        self.id2category = {v: k for k, v in self.category2id.items()}


        self.transform = transform
        self.resize_size = resize_size
        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]
        self.train = is_train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.imgs_path, path)).convert('RGB')
        img = np.array(img)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = np.array([o['bbox'] for o in ann], dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0] > 0:
            boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
            boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1e-5)  # xmax ≥ xmin + tiny value
            boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1e-5)  # ymax ≥ ymin + tiny value
            # Validate and filter boxes


        if self.train and random.random() < 0.5:
            img, boxes = flip(Image.fromarray(img), boxes)
            img = np.array(img)

        img, boxes , scale  = preprocess_img_boxes(img, boxes, self.resize_size)

        classes = [self.category2id[o['category_id']] for o in ann]
        classes = np.array(classes, dtype=np.int64).reshape(-1)

        # Convert to Jittor vars
        classes = jt.array(classes, dtype=jt.int64) if len(classes) > 0 else jt.ones([0], dtype=jt.int64)
        boxes = jt.array(boxes, dtype=jt.float32) if boxes.shape[0] > 0 else jt.ones([0, 4], dtype=jt.float32)
        img = jt.array(img, dtype=jt.float32) / 255.0

        # Debug check: make sure classes are in range
        if classes.numel() > 0:
            if classes.max().item() >= DefaultConfig.class_num or classes.min().item() < 0:
                print("[WARNING] Bad class index found:",
                       classes.numpy().tolist(),
                       f"class_num={DefaultConfig.class_num}")

        return img, boxes, classes ,scale

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0: return False
        if self._has_only_empty_bbox(annot): return False
        return True

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list ,scale_list = zip(*data)

        # Pad images
        max_h = max([img.shape[1] for img in imgs_list])
        max_w = max([img.shape[2] for img in imgs_list])
        pad_imgs = []
        for img in imgs_list:
            _, h, w = img.shape
            padded = jt.nn.pad(img, (0, max_w - w, 0, max_h - h), mode="constant", value=0.)
            pad_imgs.append(transforms.Normalize(self.mean, self.std)(padded))
        batch_imgs = jt.stack(pad_imgs)

        # Pad boxes and classes
        max_num = max([b.shape[0] for b in boxes_list]) if boxes_list else 0
        pad_boxes, pad_classes = [], []
        for boxes, classes in zip(boxes_list, classes_list):
            if boxes.shape[0] == 0:
                # no gt case
                boxes = jt.ones([1, 4]) * -1
                classes = jt.ones([1]) * -1
            if max_num > 0:
                pad_boxes.append(jt.nn.pad(boxes, (0, 0, 0, max_num - boxes.shape[0]),
                                           mode="constant", value=-1))
                pad_classes.append(jt.nn.pad(classes, (0, max_num - classes.shape[0]),
                                             mode="constant", value=-1))
            else:
                pad_boxes.append(boxes)
                pad_classes.append(classes)
        batch_boxes = jt.stack(pad_boxes)
        batch_classes = jt.stack(pad_classes)
        batch_scales = jt.array(scale_list, dtype=jt.float32).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        
        return batch_imgs, batch_boxes, batch_classes, batch_scales
