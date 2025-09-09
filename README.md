# FCOS implementation with Jittor<br>
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
## Sructure of Project





