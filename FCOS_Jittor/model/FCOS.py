import jittor as jt
import jittor.nn as nn
import numpy as np
from .head import ClsCntRegHead
from .fpn import FPN
from .resnet import resnet50
from .loss import GenTargets, LOSS, coords_fmap2orig
from .config import DefaultConfig
import math
""" from head import ClsCntRegHead
from fpn import FPN
from resnet import resnet50
from loss import GenTargets, LOSS, coords_fmap2orig
from config import DefaultConfig
import math  """

class FCOS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet50(pretrained=config.pretrained, if_include_top=False)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels, config.class_num,
                                  config.use_GN_head, config.cnt_on_reg, config.prior)
        self.config = config

    #def train(self, mode=True):
    #    super().train(mode=True)
    #    def freeze_bn(module):
    #        if isinstance(module, nn.BatchNorm):
    #            module.eval()
    #        classname = module.__class__.__name__
    #        if classname.find('BatchNorm') != -1:
    #            for p in module.parameters():
    #                p.requires_grad = False
    #
    #    if self.config.freeze_bn:
    #        self.apply(freeze_bn)
    #        print("INFO===>success frozen BN")
    #    if self.config.freeze_stage_1:
    #        self.backbone.freeze_stages(1)
    #       print("INFO===>success frozen backbone stage1")

    def execute(self, x):
        C3, C4, C5 = self.backbone(x)

        # --- ADD THIS DEBUGGING BLOCK ---
        print(f"[FCOS] Backbone output shapes -> C3: {C3.shape}, C4: {C4.shape}, C5: {C5.shape}")
        # --- END OF DEBUGGING BLOCK ---

        all_P = self.fpn([C3, C4, C5])
        
        # --- ADD THIS DEBUGGING BLOCK ---
        print(f"[FCOS] FPN output shapes -> P3-P7: {[p.shape for p in all_P]}")
        # --- END OF DEBUGGING BLOCK ---

        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config


    def execute(self, inputs):
        # Unpack the lists of feature maps from the FCOS head
        cls_logits_list, cnt_logits_list, reg_preds_list = inputs

        # This logic ensures coordinates are generated from the SAME source
        # as the regression predictions, which is the correct workaround.
        reg_preds, coords = self._reshape_cat_out(reg_preds_list, self.strides)
        cls_logits, _ = self._reshape_cat_out(cls_logits_list, self.strides)
        cnt_logits, _ = self._reshape_cat_out(cnt_logits_list, self.strides)

        # --- The rest of the function proceeds as normal ---
        cls_preds = cls_logits.sigmoid()
        cnt_preds = cnt_logits.sigmoid()

        # In model/FCOS.py
        cls_scores = jt.max(cls_preds, dim=-1)
        cls_classes = jt.argmax(cls_preds, dim=-1)[0]

        if self.config.add_centerness:
            cls_scores = jt.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))

        cls_classes = cls_classes + 1

        boxes = self._coords2boxes(coords, reg_preds)

        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk = jt.topk(cls_scores, k=max_num, dim=-1, largest=True, sorted=True)
        topk_ind = topk[1]

        _cls_scores = []
        _cls_classes = []
        _boxes = []
        
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])
            _boxes.append(boxes[batch][topk_ind[batch]])
        
        cls_scores_topk = jt.stack(_cls_scores, dim=0)
        cls_classes_topk = jt.stack(_cls_classes, dim=0)
        boxes_topk = jt.stack(_boxes, dim=0)
        
        assert boxes_topk.shape[-1] == 4

        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])



    def _post_process(self, preds_topk):
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]
            _cls_classes_b = cls_classes_topk[batch][mask]
            _boxes_b = boxes_topk[batch][mask]
            
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
            
        scores, classes, boxes = jt.stack(_cls_scores_post, dim=0), jt.stack(_cls_classes_post, dim=0), jt.stack(_boxes_post, dim=0)
        
        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        if boxes.shape[0] == 0:
            return jt.empty(shape=(0,), dtype=jt.int64)
        
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        sort_result = jt.argsort(scores, descending=True)
        order = sort_result[1] if isinstance(sort_result, tuple) else sort_result
        
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)
                
            xmin = jt.maximum(x1[order[1:]], x1[i])
            ymin = jt.maximum(y1[order[1:]], y1[i])
            xmax = jt.minimum(x2[order[1:]], x2[i])
            ymax = jt.minimum(y2[order[1:]], y2[i])
            
            inter = (xmax - xmin).clamp(min_v=0) * (ymax - ymin).clamp(min_v=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # --- START OF FIX ---
            # 1. Get indices from .nonzero() without squeezing immediately.
            nonzero_idx = (iou <= thr).nonzero()

            # 2. Check if the result is empty BEFORE squeezing.
            if nonzero_idx.numel() == 0:
                break
            
            # 3. Now it's safe to squeeze. Use squeeze(1) to ensure it's always a 1D tensor.
            #    This also simplifies the logic below.
            idx = nonzero_idx.squeeze(1)

            # 4. Update order. The previous check for idx.ndim is no longer needed.
            order = order[idx + 1]
            # --- END OF FIX ---

        return jt.array(keep, dtype=jt.int64)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):
        if boxes.numel() == 0:
            return jt.empty(shape=(0,), dtype=jt.int64)

        max_coordinate = jt.max(boxes)
        offsets = idxs.float32() * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets.unsqueeze(1)
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        # unsqueeze coords to add a batch dimension for broadcasting
        x1y1 = coords.unsqueeze(0) - offsets[..., :2]
        x2y2 = coords.unsqueeze(0) + offsets[..., 2:]
        boxes = jt.concat([x1y1, x2y2], dim=-1)
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            # FIX: Generate coordinates from the original [B, C, H, W] shape FIRST.
            coord = coords_fmap2orig(pred, stride)
            
            # THEN, permute and reshape the prediction tensor for concatenation.
            pred = pred.permute(0, 2, 3, 1)
            pred = jt.reshape(pred, [batch_size, -1, c])
            
            out.append(pred)
            coords.append(coord)
        return jt.concat(out, dim=1), jt.concat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp(min_v=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[:, [0, 2]] = batch_boxes[:, [0, 2]].clamp(max_v=w - 1)
        batch_boxes[:, [1, 3]] = batch_boxes[:, [1, 3]].clamp(max_v=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, mode="training", config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.mode = mode
        self.fcos_body = FCOS(config=config)

        if mode == "training":
            self.target_layer = GenTargets(strides=config.strides, limit_range=config.limit_range)
            self.loss_layer = LOSS(config=config)
            # --- ADD THE FREEZING LOGIC HERE ---
            if config.freeze_bn:
                for module in self.fcos_body.modules():
                    if isinstance(module, nn.BatchNorm):
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
                print("INFO===>success frozen BN")
            if config.freeze_stage_1:
                self.fcos_body.backbone.freeze_stages(1)
                print("INFO===>success frozen backbone stage1")
            # --- END OF ADDED BLOCK ---
        elif mode == "inference":
            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                             config.max_detection_boxes_num, config.strides, config)
            self.clip_boxes = ClipBoxes()

    def execute(self, inputs):
        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes, batch_scales = inputs
            out = self.fcos_body(batch_imgs)
            # NEW, CORRECTED CODE
            out = self.fcos_body(batch_imgs)

            # The target_layer now returns a single tuple/list with 4 items
            targets_with_coords = self.target_layer([out, batch_boxes, batch_classes, batch_scales])

            # The loss_layer expects a list of [predictions, targets_with_coords]
            losses = self.loss_layer(out, targets_with_coords)
            return losses
        elif self.mode == "inference":
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes

#--test--






if __name__ == '__main__':
    # This main block is for testing and comparing the full FCOSDetector against a PyTorch implementation.
    
    # Import PyTorch
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Skipping comparison test.")
        exit()
    
    # --- START: Self-Contained PyTorch Implementation ---
    
    class BottleneckPytorch(torch.nn.Module):
        expansion = 4
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(planes)
            self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(planes)
            self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = torch.nn.BatchNorm2d(planes * 4)
            self.relu = torch.nn.ReLU(inplace=True)
            self.downsample = downsample
        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class MockResNet50Pytorch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = torch.nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU(inplace=True)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(BottleneckPytorch, 64, 3)
            self.layer2 = self._make_layer(BottleneckPytorch, 128, 4, stride=2)
            self.layer3 = self._make_layer(BottleneckPytorch, 256, 6, stride=2)
            self.layer4 = self._make_layer(BottleneckPytorch, 512, 3, stride=2)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(planes * block.expansion)
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return torch.nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            c3 = self.layer2(x) # Output features are from layer2, not x
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            return c3, c4, c5
    
    # Mock PyTorch FPN
    class MockFPNPytorch(torch.nn.Module):
        def __init__(self, out_channels, use_p5):
            super().__init__()
            self.in_channels = [512, 1024, 2048]
            self.out_channels = out_channels
            self.use_p5 = use_p5
            self.lat_convs = torch.nn.ModuleList([torch.nn.Conv2d(c, out_channels, 1) for c in self.in_channels])
            self.fpn_convs = torch.nn.ModuleList([torch.nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in self.in_channels])
            
            # --- START OF FIX ---
            # 1. P6's input channel is now `out_channels` (from P5), not 2048 (from C5).
            self.p6 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            # --- END OF FIX ---

            self.p7 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        def forward(self, inputs):
            c3, c4, c5 = inputs
            
            # Lateral connections
            p5_lat = self.lat_convs[2](c5)
            p4_lat = self.lat_convs[1](c4)
            p3_lat = self.lat_convs[0](c3)
            
            # Top-down pathway
            p4 = p4_lat + torch.nn.functional.interpolate(p5_lat, size=c4.shape[-2:], mode='nearest')
            p3 = p3_lat + torch.nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
            
            # Output convolutions
            p3 = self.fpn_convs[0](p3)
            p4 = self.fpn_convs[1](p4)
            p5 = self.fpn_convs[2](p5_lat) # Use p5_lat here
            
            # --- START OF FIX ---
            # 2. P6 now takes p5 as input, not c5.
            p6 = self.p6(p5)
            # --- END OF FIX ---
            
            p7 = self.p7(torch.nn.functional.relu(p6))
            
            return [p3, p4, p5, p6, p7]

    # Use the self-contained PyTorch Head
    class ScaleExpPytorch(torch.nn.Module):
        def __init__(self,init_value=1.0):
            super().__init__()
            self.scale=torch.nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
        def forward(self,x):
            return torch.exp(x*self.scale)

    class ClsCntRegHeadPytorch(torch.nn.Module):
        def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
            super().__init__()
            self.cnt_on_reg=cnt_on_reg
            cls_branch, reg_branch = [], []
            for i in range(4):
                cls_branch.extend([torch.nn.Conv2d(in_channel,in_channel,3,padding=1,bias=True), torch.nn.GroupNorm(32,in_channel) if GN else torch.nn.Identity(), torch.nn.ReLU(True)])
                reg_branch.extend([torch.nn.Conv2d(in_channel,in_channel,3,padding=1,bias=True), torch.nn.GroupNorm(32,in_channel) if GN else torch.nn.Identity(), torch.nn.ReLU(True)])
            self.cls_conv=torch.nn.Sequential(*cls_branch)
            self.reg_conv=torch.nn.Sequential(*reg_branch)
            self.cls_logits=torch.nn.Conv2d(in_channel,class_num,3,padding=1)
            self.cnt_logits=torch.nn.Conv2d(in_channel,1,3,padding=1)
            self.reg_pred=torch.nn.Conv2d(in_channel,4,3,padding=1)
            self.scale_exp = torch.nn.ModuleList([ScaleExpPytorch(1.0) for _ in range(5)])

        def forward(self,inputs):
            cls_logits, cnt_logits, reg_preds = [], [], []
            for i, P in enumerate(inputs):
                cls_out, reg_out = self.cls_conv(P), self.reg_conv(P)
                cls_logits.append(self.cls_logits(cls_out))
                cnt_logits.append(self.cnt_logits(reg_out if self.cnt_on_reg else cls_out))
                reg_preds.append(self.scale_exp[i](self.reg_pred(reg_out)))
            return cls_logits, cnt_logits, reg_preds

    # PyTorch FCOS Body
    class FCOSPytorch(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.backbone = MockResNet50Pytorch()
            self.fpn = MockFPNPytorch(config.fpn_out_channels, config.use_p5)
            self.head = ClsCntRegHeadPytorch(config.fpn_out_channels, config.class_num, config.use_GN_head, config.cnt_on_reg, config.prior)
        def forward(self, x):
            c3, c4, c5 = self.backbone(x)
            all_p = self.fpn([c3, c4, c5])
            return self.head(all_p)

    # PyTorch Detector
    class FCOSDetectorPytorch(torch.nn.Module):
        def __init__(self, mode="inference", config=None):
            super().__init__()
            self.fcos_body = FCOSPytorch(config=config)
        def forward(self, inputs):
            return self.fcos_body(inputs)

    # --- END: Self-Contained PyTorch Implementation ---

    print("--- Starting Jittor vs. PyTorch FCOSDetector Comparison ---")
    
    # Setup
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    
    config = DefaultConfig
    config.pretrained = False

    # --- 1. Create Identical Mock Input Data ---
    print("\nCreating mock image input...")
    mock_image_np = np.random.randn(1, 3, 800, 800).astype('float32')
    mock_image_jt = jt.array(mock_image_np)
    mock_image_pt = torch.from_numpy(mock_image_np)

    # --- 2. Instantiate Models ---
    print("\nInstantiating models...")
    jittor_detector = FCOSDetector(mode="inference", config=config)
    pytorch_detector = FCOSDetectorPytorch(mode="inference", config=config)
    jittor_detector.eval()
    pytorch_detector.eval()

    # --- 3. Copy Weights from PyTorch to Jittor ---
    print("\nCopying weights...")
    pt_params = dict(pytorch_detector.named_parameters())
    for name, jt_param in jittor_detector.named_parameters():
        if name in pt_params:
            pt_param = pt_params[name]
            if list(jt_param.shape) == list(pt_param.shape):
                 jt_param.assign(pt_param.detach().cpu().numpy())
    print("Weight copy complete.")

    # --- 4. Run Inference ---
    print("\nRunning inference...")
    with jt.no_grad():
        cls_jt, cnt_jt, reg_jt = jittor_detector.fcos_body(mock_image_jt)

    with torch.no_grad():
        cls_pt, cnt_pt, reg_pt = pytorch_detector.fcos_body(mock_image_pt)
    print("Inference complete.")

    # --- 5. Compare Outputs ---
    print("\n--- Comparing Raw Model Outputs ---")
    all_match = True
    def compare_tensors(name, jt_tensor, pt_tensor, atol=1e-4):
        global all_match
        jt_np = jt_tensor.numpy()
        pt_np = pt_tensor.detach().cpu().numpy()
        if np.allclose(jt_np, pt_np, atol=atol):
            print(f"✅ {name}: Outputs MATCH")
        else:
            all_match = False
            diff = np.abs(jt_np - pt_np).max()
            print(f"❌ {name}: Outputs DO NOT MATCH (max difference: {diff})")

    for i in range(len(cls_jt)):
        print(f"\n--- FPN Level P{i+3} ---")
        compare_tensors(f"Cls Logits", cls_jt[i], cls_pt[i])
        compare_tensors(f"Cnt Logits", cnt_jt[i], cnt_pt[i])
        compare_tensors(f"Reg Preds", reg_jt[i], reg_pt[i])

    print("\n--- Final Result ---")
    if all_match:
        print("🎉🎉🎉 All Jittor and PyTorch raw outputs are identical.")
    else:
        print("🔥🔥🔥 Mismatch detected in raw model outputs.")






























