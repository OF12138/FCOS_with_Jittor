# ATSS-FCOS implementation with Jittor<br>
Translate the FCOS_pytorch_version to Jittor_version<br>
The project provide two versions of FCOS implementation.<br>

## Environment Setup<br>

### Jittor version<br>
-OS: Ubuntu(WSL) 24.04.2 LTS<br>
-Python: 3.8.20<br>
-CUDA: 12.1<br>
-Jittor: 1.3.9.14<br>
-GCC: 11.5.0<br>

### Pytorch version<br>
-Pytorch: 2.4.1<br>
-Others: the same with jittor environment<br>

## preparation
You should download the dataset to  "FCOS_Jittor/dataset/tiny_coco"<br>
[tiny_coco dataset](https://www.kaggle.com/datasets/weipengchao/tiny-coco1k) <br>
You should download the resnet50-19c8e357.pth to "FCOS_Jittor/model/resnet50-19c8e357.pth"<br>
Or you can change the pretrained parameter from True to False in config.py<br>

## Check List
-[x] Could be trained <br>
-[x] Loss Could decrease successfully<br>
-[x] Could Detect<br>
-[ ] Could detect correctly<br>

## Structure of Project<br>
ATSS-FCOS in Jittor/Pytorch<br>
├── dataset<br>                    
│   ├── augmentation.py             //to preprocess the images<br>
│   └── dataset.py<br>
├── model                           //this is the folder where backbone net locates<br>
│   ├── config.py<br>
│   ├── FCOS.py<br>                     
│   ├── fpn.py<br>
│   ├── head.py<br>
│   ├── loss.py                     //the ATSS trick is applied here<br>
│   └── resnet.py<br>
├── detect.py                       //object detection<br>
└── train.py<br>

## Result Display<br>
### Loss in tensorboard<br>









