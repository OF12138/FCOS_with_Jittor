import jittor as jt
from jittor.dataset import Dataset
import jittor.transform as transforms
from pycocotools.coco import COCO
import numpy as np
import cv2
from PIL import Image
import random
import os
import json

class DefaultConfig():
    #backbone
    pretrained=True
    freeze_stage_1= True
    freeze_bn= True
    #fpn
    fpn_out_channels= 256
    use_p5=True
    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True
    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    # Data processing
    pixel_mean = [0.40789654, 0.44719302, 0.47026115]
    pixel_std = [0.28863828, 0.27408164, 0.27809835]
    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000

# -----------------------------
# Utils
# -----------------------------

def image_normalize(img, mean, std):
    """
    Function for normalizing an image tensor.
    """
    if not isinstance(img, jt.Var):
        raise TypeError(f'Input type should be jt.Var. Got {type(img)}.')
    
    mean = jt.array(mean).reshape(-1, 1, 1)
    std = jt.array(std).reshape(-1, 1, 1)
    
    if (std.data == 0).any():
        raise ValueError('std cannot be zero.')
        
    return (img - mean) / std

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
    """
    Resizes and pads image while correctly transforming bounding boxes.
    """
    min_side, max_side = input_size
    h, w, _ = image.shape
    scale = min(max_side / h, max_side / w) if h > 0 and w > 0 else 1.0
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full((max_side, max_side, 3), 0, dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    if boxes.shape[0] > 0:
        boxes[:, :4] *= scale
    return image_paded, boxes, scale

# -----------------------------
# Dataset
# -----------------------------
class COCODataset(Dataset):
    CLASSES_NAME = ('__back_ground__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, imgs_path, anno_path, resize_size=[800,1333], is_train=True, transform=None):
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
        cat_ids = sorted(self.coco.getCatIds())
        self.category2id = {v: i+1 for i, v in enumerate(cat_ids)}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.transform = transform
        self.resize_size = resize_size
        self.mean = DefaultConfig.pixel_mean
        self.std = DefaultConfig.pixel_std
        self.train = is_train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.imgs_path, path)).convert('RGB')
        
        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = np.array([o['bbox'] for o in ann], dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0] > 0:
            boxes[:, 2:] += boxes[:, :2]

        if self.train and random.random() < 0.5:
            img, boxes = flip(img, boxes)
        
        img = np.array(img)
        img, boxes, scale = preprocess_img_boxes(img, boxes, self.resize_size)
        
        classes = [self.category2id[o['category_id']] for o in ann]
        classes = np.array(classes, dtype=np.int64).reshape(-1)

        # FIX: Convert numpy image to Jittor Var to resolve the error.
        # This function converts an (H, W, C) numpy array to a (C, H, W) Jittor Var
        # and scales pixel values from [0, 255] to [0.0, 1.0].
        img = img.transpose((2, 0, 1))
        img = transforms.to_tensor(img) 
        
        boxes = jt.array(boxes) if boxes.shape[0] > 0 else jt.zeros([0, 4])
        classes = jt.array(classes) if len(classes) > 0 else jt.zeros([0], 'int64')
        scale = jt.array(scale)
        
        return img, boxes, classes, scale

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0: return False
        if self._has_only_empty_bbox(annot): return False
        return True

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, scale_list = zip(*data)
        
        batch_size = len(imgs_list)
        max_h = max([s.shape[1] for s in imgs_list])
        max_w = max([s.shape[2] for s in imgs_list])
        pad_imgs_list = []
        for i in range(batch_size):
            img = imgs_list[i]
            img=jt.array(img)
            padded_img = jt.nn.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]), value=0.)
            # Use the corrected image_normalize function
            norm_img = image_normalize(padded_img, self.mean, self.std)
            pad_imgs_list.append(norm_img)
        batch_imgs = jt.stack(pad_imgs_list)
        
        max_num = max([b.shape[0] for b in boxes_list]) if boxes_list else 0
        pad_boxes_list, pad_classes_list = [], []
        if max_num > 0:
            for i in range(batch_size):
                pad_boxes_list.append(jt.nn.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
                pad_classes_list.append(jt.nn.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))
        else:
            for _ in range(batch_size):
                pad_boxes_list.append(jt.ones([1, 4]) * -1)
                pad_classes_list.append(jt.ones([1]) * -1)

        batch_boxes = jt.stack(pad_boxes_list)
        batch_classes = jt.stack(pad_classes_list)
        batch_scales = jt.stack(scale_list).unsqueeze(-1)
        
        return batch_imgs, batch_boxes, batch_classes, batch_scales

if __name__ == '__main__':
    try:
        import torch
        from torchvision.datasets import CocoDetection
        from torchvision import transforms as tv_transforms
    except ImportError:
        print("PyTorch or Torchvision not found. Skipping comparison test.")
        exit()

    # --- START: Self-Contained PyTorch Implementation ---
    def flip_pt(img, boxes):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        if boxes.shape[0] != 0:
            xmin = w - boxes[:,2]; xmax = w - boxes[:,0]
            boxes[:, 0], boxes[:, 2] = xmin, xmax
        return img, boxes

    class COCODatasetPytorch(CocoDetection):
        def __init__(self,imgs_path,anno_path,resize_size=[800,1333],is_train=True):
            super().__init__(imgs_path,anno_path)
            ids=[]
            for id in self.ids:
                ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)
                ann=self.coco.loadAnns(ann_id)
                if self._has_valid_annotation(ann): ids.append(id)
            self.ids=ids
            cat_ids = sorted(self.coco.getCatIds())
            self.category2id = {v: i + 1 for i, v in enumerate(cat_ids)}
            self.resize_size=resize_size
            self.mean=DefaultConfig.pixel_mean
            self.std=DefaultConfig.pixel_std
            self.train = is_train
            
        def __getitem__(self,index):
            img,ann=super().__getitem__(index)
            ann = [o for o in ann if o['iscrowd'] == 0]
            boxes = np.array([o['bbox'] for o in ann], dtype=np.float32).reshape(-1, 4)
            if boxes.shape[0] > 0: boxes[...,2:] += boxes[...,:2]
            if self.train and random.random() < 0.5: img, boxes = flip_pt(img, boxes)
            img=np.array(img)
            img, boxes, scale = self.preprocess_img_boxes(img,boxes,self.resize_size)
            classes = [self.category2id[o['category_id']] for o in ann]
            img=tv_transforms.ToTensor()(img)
            boxes=torch.from_numpy(boxes)
            classes=torch.LongTensor(classes)
            scale=torch.tensor(scale, dtype=torch.float32)
            return img,boxes,classes,scale

        def preprocess_img_boxes(self,image,boxes,input_ksize):
            min_side, max_side = input_ksize
            h, w, _ = image.shape
            scale = min(max_side / h, max_side / w) if h > 0 and w > 0 else 1.0
            nh, nw = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (nw, nh))
            image_paded = np.full((max_side, max_side, 3), 0, dtype=np.uint8)
            image_paded[:nh, :nw, :] = image_resized
            if boxes.shape[0] > 0: boxes[:, :4] *= scale
            return image_paded, boxes, scale

        def _has_valid_annotation(self,annot):
            if len(annot) == 0: return False
            if all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot): return False
            return True

        def collate_fn(self,data):
            imgs_list,boxes_list,classes_list,scale_list=zip(*data)
            batch_size=len(boxes_list)
            pad_imgs_list, pad_boxes_list, pad_classes_list = [], [], []
            max_h = max([int(s.shape[1]) for s in imgs_list])
            max_w = max([int(s.shape[2]) for s in imgs_list])
            for img in imgs_list:
                padded = torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)
                pad_imgs_list.append(tv_transforms.functional.normalize(padded,self.mean,self.std,inplace=False))
            max_num=max([b.shape[0] for b in boxes_list]) if boxes_list else 0
            if max_num > 0:
                for i in range(batch_size):
                    pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
                    pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
            else:
                for _ in range(batch_size):
                    pad_boxes_list.append(torch.ones((1, 4)) * -1)
                    pad_classes_list.append(torch.ones((1)) * -1)
            batch_boxes=torch.stack(pad_boxes_list)
            batch_classes=torch.stack(pad_classes_list)
            batch_imgs=torch.stack(pad_imgs_list)
            batch_scales=torch.stack(scale_list).unsqueeze(-1)
            return batch_imgs,batch_boxes,batch_classes,batch_scales
    # --- END: Self-Contained PyTorch Implementation ---

    # --- Create Mock Data & Run Comparison ---
    mock_img_dir = "test_imgs"; mock_anno_file = "test_anno.json"
    os.makedirs(mock_img_dir, exist_ok=True)
    Image.new('RGB', (640, 480)).save(os.path.join(mock_img_dir, "test_img.jpg"))
    mock_anno = {"images": [{"id": 1, "file_name": "test_img.jpg", "height": 480, "width": 640}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10,10,50,50], "area": 2500, "iscrowd": 0}], "categories": [{"id": 1, "name": "person"}]}
    with open(mock_anno_file, 'w') as f: json.dump(mock_anno, f)
    
    print("--- Initializing Datasets ---")
    dataset_jt = COCODataset(mock_img_dir, mock_anno_file, resize_size=[800, 800], is_train=False)
    dataset_pt = COCODatasetPytorch(mock_img_dir, mock_anno_file, resize_size=[800, 800], is_train=False)

    def compare_outputs(name, jt_out, pt_out):
        # Handle both jt.Var and potential raw types
        jt_np = jt_out.numpy() if hasattr(jt_out, 'numpy') else np.array(jt_out)
        pt_np = pt_out.detach().cpu().numpy()
        print(f"\n--- {name} ---")
        print(f"Jittor shape: {jt_np.shape}, dtype: {jt_np.dtype}")
        print(f"PyTorch shape: {pt_np.shape}, dtype: {pt_np.dtype}")
        if np.allclose(jt_np, pt_np, atol=1e-5): print(f"✅ Values MATCH")
        else: print(f"❌ Values DO NOT MATCH (max diff: {np.abs(jt_np - pt_np).max()})")

    print("\n--- Comparing Single Item Output (__getitem__) ---")
    random.seed(0); jt.set_global_seed(0); torch.manual_seed(0)
    img_jt, boxes_jt, classes_jt, scale_jt = dataset_jt[0]
    random.seed(0); jt.set_global_seed(0); torch.manual_seed(0)
    img_pt, boxes_pt, classes_pt, scale_pt = dataset_pt[0]
    compare_outputs("Image", img_jt, img_pt)
    compare_outputs("Boxes", boxes_jt, boxes_pt)
    compare_outputs("Classes", classes_jt, classes_pt)
    compare_outputs("Scale", scale_jt, scale_pt)

    print("\n\n--- Comparing Batched Output (collate_fn) ---")
    batch_jt = dataset_jt.collate_fn([dataset_jt[0], dataset_jt[0]])
    batch_pt = dataset_pt.collate_fn([dataset_pt[0], dataset_pt[0]])
    compare_outputs("Batched Images", batch_jt[0], batch_pt[0])
    compare_outputs("Batched Boxes", batch_jt[1], batch_pt[1])
    compare_outputs("Batched Classes", batch_jt[2], batch_pt[2])
    compare_outputs("Batched Scales", batch_jt[3], batch_pt[3])

    os.remove(os.path.join(mock_img_dir, "test_img.jpg")); os.rmdir(mock_img_dir); os.remove(mock_anno_file)

