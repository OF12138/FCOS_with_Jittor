import jittor as jt
import jittor.nn as nn
from .head import ClsCntRegHead
from .fpn import FPN
from .resnet import resnet50
from .loss import GenTargets, LOSS, coords_fmap2orig
from .config import DefaultConfig


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

    def train(self, mode=True):
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def execute(self, x):
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
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
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)

        cls_preds = cls_logits.sigmoid()
        cnt_preds = cnt_logits.sigmoid()
        
        # Jittor tensors are on the GPU automatically if a GPU is available, so the check is not needed.
        # coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = jt.max(cls_preds, dim=-1)

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
        order = jt.argsort(scores, descending=True)
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
            
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            
            # The indexing is different in Jittor; jt.array() is used to create a new tensor
            # The next 'order' are the remaining indices.
            if idx.ndim == 0:
                order = order[idx + 1].unsqueeze(0)
            else:
                order = order[idx + 1]

        return jt.array(keep, dtype=jt.int64)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):
        if boxes.numel() == 0:
            return jt.empty(shape=(0,), dtype=jt.int64)

        max_coordinate = jt.max(boxes)
        # Jittor needs explicit casting for operations
        offsets = idxs.float32() * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets.unsqueeze(1)
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
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
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride)
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
            self.loss_layer = LOSS()
        elif mode == "inference":
            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                             config.max_detection_boxes_num, config.strides, config)
            self.clip_boxes = ClipBoxes()

    def execute(self, inputs):
        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer([out, batch_boxes, batch_classes])
            losses = self.loss_layer([out, targets])
            return losses
        elif self.mode == "inference":
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes