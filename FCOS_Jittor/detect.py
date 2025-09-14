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
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side = input_ksize
    h, w, _ = image.shape

    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w = 32 - nw % 32
    pad_h = 32 - nh % 32

    image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded

if __name__ == "__main__":
    jt.flags.use_cuda = 1  # Use GPU if available

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]

    class Config:
        # backbone
        pretrained = False
        freeze_stage_1 = True
        freeze_bn = True

        # fpn
        fpn_out_channels = 256
        use_p5 = True

        # head
        class_num = 80
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = False

        # training
        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

        # inference
        score_threshold = 0.3
        nms_iou_threshold = 0.4
        max_detection_boxes_num = 300

    model = FCOSDetector(mode="inference", config=Config)
    
    # Load state dict
    #state_dict = jt.load("./checkpoint/model_4.pth")
    #model.load_state_dict(state_dict)
    model.load("./checkpoint/model_5.pth")
    
    # Set to evaluation mode
    model.eval()
    print("===>success loading model")

    # Image inference and processing
    root = "./test_images/"
    if not os.path.exists('out_images'):
        os.makedirs('out_images')
        
    names = os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        img_pad = preprocess_img(img_bgr, [800, 1333])
        img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        
        # Convert to Jittor tensor
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = jt.array(transforms.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor))
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(dim=0)

        # Inference
        start_t = time.time()
        with jt.no_grad():
            out = model(img_tensor)
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)
        
        scores, classes, boxes = out

        # Convert Jittor tensors to NumPy arrays for visualization
        boxes = boxes[0].numpy().tolist()
        classes = classes[0].numpy().tolist()
        scores = scores[0].numpy().tolist()

        # Visualization
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for i, box in enumerate(boxes):
            if scores[i] > 0.3:  # Only draw boxes with confidence above a certain threshold
                print(f"  [DETECT] Box for class '{COCODataset.CLASSES_NAME[int(classes[i])]}' with score {scores[i]:.3f}: {np.array(box).astype(int)}")
                
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                
                b_color = colors[int(classes[i]) - 1]
                #bbox = patches.Rectangle((box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], linewidth=1, facecolor='none', edgecolor=b_color)
                bbox = patches.Rectangle(
                    (box[0], box[1]), 
                    width=box[2] - box[0], 
                    height=box[3] - box[1], 
                    linewidth=2,  # Increased linewidth for visibility
                    facecolor='none', 
                    edgecolor=b_color
                )
                ax.add_patch(bbox)

                plt.text(
                    box[0] + 5,  # Add a small x-offset
                    box[1] + 20, # Add a small y-offset for text visibility
                    s=f"{COCODataset.CLASSES_NAME[int(classes[i])]} {scores[i]:.3f}", 
                    color='white',
                    verticalalignment='top',
                    bbox={'color': b_color, 'pad': 0}
                )
                         
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('out_images/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
        plt.close()