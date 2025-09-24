class DefaultConfig():
    #backbone
    pretrained=True #### should be true
    freeze_stage_1= True
    freeze_bn= True

    #fpn
    fpn_out_channels= 256# should be 256
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
    # Mean and std for normalization, must be consistent for training and inference
    pixel_mean = [0.40789654, 0.44719302, 0.47026115]
    pixel_std = [0.28863828, 0.27408164, 0.27809835]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000