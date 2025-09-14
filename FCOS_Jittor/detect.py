import cv2
from model.FCOS import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.dataset import COCODataset
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from collections import OrderedDict

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    #requires input images to have dimensions that are a multiple of 32
    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    
#convert multiple GPUs BN to single GPU BN
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

#ensure the script only runs when executed directly
if __name__=="__main__":

    #set the palette
    #cmap = plt.get_cmap('tab20b')
    #colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    cmap = plt.get_cmap('hsv') # or another colormap with enough colors
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]

    #set the configuration
    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        class_num=80 #tiny_coco dataset
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=False

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

        #inference
        score_threshold=0.3
        nms_iou_threshold=0.4
        max_detection_boxes_num=300

    model=FCOSDetector(mode="inference",config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")

    model = torch.nn.DataParallel(model)

    # Load the checkpoint
    device = torch.device('cpu')
    state_dict = torch.load("./checkpoint/model_1.pth", map_location=device, weights_only=True)

    # Create a new state_dict with the 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  # add `module.` prefix
        new_state_dict[name] = v

    # Load the new state_dict
    model.load_state_dict(new_state_dict)
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")

    #start the inference mode
    model=model.eval()
    print("===>success loading model")

    #image inference and processing
    import os
    root="./test_images/"
    names=os.listdir(root)
    for name in names:

        #get the image and preprocessing
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[800,1333])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img1=transforms.ToTensor()(img)
        img1= transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225],inplace=True)(img1)
        img1=img1
        
        #inference
        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        #visualization
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img) #display the original image

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0)) #draw the bounding box
            b_color = colors[int(classes[i]) - 1]
            bbox = patches.Rectangle((box[0],box[1]),width=box[2]-box[0],height=box[3]-box[1],linewidth=1,facecolor='none',edgecolor=b_color)
            ax.add_patch(bbox)

            #The class label and confidence score are added as text next to each bounding box.
            plt.text(box[0], box[1], s="%s %.3f"%(COCODataset.CLASSES_NAME[int(classes[i])],scores[i]), color='white',
                     verticalalignment='top',
                     bbox={'color': b_color, 'pad': 0})

        #save the output image
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('out_images/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
        plt.close()