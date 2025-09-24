# detect.py (FINAL CORRECTED VERSION)
import cv2
from model.FCOS import FCOSDetector
import jittor as jt
from jittor import transform as transforms
import numpy as np
from dataset.dataset import COCODataset
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import os


def preprocess_img(image, input_ksize):
    """
    Resizes and pads image according to the training logic.
    """
    min_side, max_side = input_ksize
    h, w, _ = image.shape

    # 1. Calculate scale to fit within the [min_side, max_side] boundary
    scale = min_side / min(h, w)
    if max(h, w) * scale > max_side:
        scale = max_side / max(h, w)

    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (nw, nh))

    # 2. Pad the image to be divisible by 32 (a common requirement for FPNs)
    pad_w = 32 - nw % 32 if nw % 32 > 0 else 0
    pad_h = 32 - nh % 32 if nh % 32 > 0 else 0

    image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized

    return image_paded, scale


if __name__ == "__main__":
    jt.flags.use_cuda = 1

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]

    class Config:
        # This config should match the one used for training
        pretrained = False
        freeze_stage_1 = True
        freeze_bn = True
        fpn_out_channels = 256
        use_p5 = True
        class_num = 80
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = True
        pixel_mean = [0.40789654, 0.44719302, 0.47026115]
        pixel_std = [0.28863828, 0.27408164, 0.27809835]
        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        score_threshold = 0.3
        nms_iou_threshold = 0.4
        max_detection_boxes_num = 300

    model = FCOSDetector(mode="inference", config=Config)
    model.load("./checkpoint/model_1.pth")
    model.eval()
    print("===>success loading model")

    root = "./test_images/"
    if not os.path.exists('out_images'):
        os.makedirs('out_images')
        
    names = os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        
        # 1. Preprocess image AND get the scale factor
        img_pad, scale = preprocess_img(img_bgr, [800, 1333]) # Use the same size as training
        
        img_rgb = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        
        img_tensor = transforms.ToTensor()(img_rgb)
        img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = jt.array(transforms.ImageNormalize(Config.pixel_mean, Config.pixel_std)(img_tensor))
        img_tensor = img_tensor.unsqueeze(dim=0)

        start_t = time.time()
        with jt.no_grad():
            scores, classes, boxes = model(img_tensor)
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)
        
        boxes = boxes[0].numpy()
        classes = classes[0].numpy()
        scores = scores[0].numpy()

        # 2. Scale boxes back to original image coordinates
        print(f"\n[DETECT] Scale factor for '{name}': {scale:.4f}")
        print(f"  [DETECT] Box on 640x640 canvas (before scaling): {np.array(boxes[0]).astype(int) if boxes.shape[0] > 0 else 'None'}")
        boxes /= scale
        print(f"  [DETECT] Box for original image (after scaling): {np.array(boxes[0]).astype(int) if boxes.shape[0] > 0 else 'None'}")

        # 3. Visualize on the ORIGINAL image (img_bgr)
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        for i, box in enumerate(boxes):
            print(f"  [DETECT] Box for class '{COCODataset.CLASSES_NAME[int(classes[i])]}' with score {scores[i]:.3f}: {np.array(box).astype(int)}")
            
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            
            b_color = colors[int(classes[i]) - 1]
            bbox = patches.Rectangle(
                (box[0], box[1]), 
                width=box[2] - box[0], 
                height=box[3] - box[1], 
                linewidth=2,
                facecolor='none', 
                edgecolor=b_color
            )
            ax.add_patch(bbox)
            plt.text(
                box[0], 
                box[1] - 5, # Place text slightly above the box
                s=f"{COCODataset.CLASSES_NAME[int(classes[i])]} {scores[i]:.2f}", 
                color='white',
                verticalalignment='top',
                bbox={'color': b_color, 'pad': 0}
            )
                         
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f'out_images/{name}', bbox_inches='tight', pad_inches=0.0)
        plt.close()