import jittor as jt
import jittor.nn as nn

# --- Helper Functions ---
def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * jt.pow((1.0 - pt), gamma) * pt.log()
    return loss.sum()

def compute_iou(boxes1, boxes2):
    """
    Computes IoU between two sets of boxes in (x1, y1, x2, y2) format.
    - boxes1: [N, 4]
    - boxes2: [M, 4]
    Returns an IoU matrix of shape [N, M].
    """
    px1, py1, px2, py2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    tx1, ty1, tx2, ty2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    ix1 = jt.maximum(px1.unsqueeze(1), tx1.unsqueeze(0))
    iy1 = jt.maximum(py1.unsqueeze(1), ty1.unsqueeze(0))
    ix2 = jt.minimum(px2.unsqueeze(1), tx2.unsqueeze(0))
    iy2 = jt.minimum(py2.unsqueeze(1), ty2.unsqueeze(0))
    
    inter_w = (ix2 - ix1).clamp(min_v=0)
    inter_h = (iy2 - iy1).clamp(min_v=0)
    overlap = inter_w * inter_h

    areas1 = (px2 - px1) * (py2 - py1)
    areas2 = (tx2 - tx1) * (ty2 - ty1)
    union = areas1.unsqueeze(1) + areas2.unsqueeze(0) - overlap
    
    return overlap / union.clamp(min_v=1e-10)

#from feature map to original full-size images
#calculate the corresponding (x,y) coordinates of each point in the original image
def coords_fmap2orig(feature, stride):
    h, w = feature.shape[-2:] #get the height and width of the feature map
    shifts_x = jt.arange(0, w, dtype=jt.float32) + 0.5 #get a list of x-coords and move to the center
    shifts_y = jt.arange(0, h, dtype=jt.float32) + 0.5
    #scale the coords by the stride
    shifts_x = shifts_x * stride
    shifts_y = shifts_y * stride
    #create a 2D grid of all (y,x) pairs
    shift_y, shift_x = jt.meshgrid(shifts_y, shifts_x)
    #flatten the grids 
    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])
    #stack them together to get a list of [x,y] coordinates
    coords = jt.stack([shift_x, shift_y], dim=-1)
    return coords

def giou_loss(preds, targets):
    """
    Calculates GIoU loss on FULL BOX COORDINATES (x1, y1, x2, y2).
    """

    # --- START: GIoU DEBUG BLOCK ---
    if preds.numel() > 0:
        print("\n    --- Inside giou_loss ---")
        print(f"    [GIOU_DBG] Input Pred Boxes (x1y1x2y2) sample:\n{preds[:2].numpy()}")
        print(f"    [GIOU_DBG] Input Target Boxes (x1y1x2y2) sample:\n{targets[:2].numpy()}")
    # --- END: GIoU DEBUG BLOCK ---

    px1, py1, px2, py2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    tx1, ty1, tx2, ty2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    pred_areas = (px2 - px1) * (py2 - py1)
    target_areas = (tx2 - tx1) * (ty2 - ty1)

    ix1 = jt.maximum(px1, tx1)
    iy1 = jt.maximum(py1, ty1)
    ix2 = jt.minimum(px2, tx2)
    iy2 = jt.minimum(py2, ty2)
    
    inter_w = (ix2 - ix1).clamp(min_v=0)
    inter_h = (iy2 - iy1).clamp(min_v=0)
    overlap = inter_w * inter_h
    
    union = pred_areas + target_areas - overlap
    iou = overlap / union.clamp(min_v=1e-10)

    # Enclosing box
    ex1 = jt.minimum(px1, tx1)
    ey1 = jt.minimum(py1, ty1)
    ex2 = jt.maximum(px2, tx2)
    ey2 = jt.maximum(py2, ty2)
    G_area = (ex2 - ex1) * (ey2 - ey1)

    giou = iou - (G_area - union) / G_area.clamp(min_v=1e-10)
    loss = 1.0 - giou


    # --- START: GIoU DEBUG BLOCK ---
    if preds.numel() > 0:
        print(f"    [GIOU_DBG] IoU (sample): {iou[:5].numpy()}")
        print(f"    [GIOU_DBG] GIoU (sample): {giou[:5].numpy()}")
    # --- END: GIoU DEBUG BLOCK ---

    return loss.sum()

# --- Main Classes ---
class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range

    def _generate_level_coords(self, h, w, stride):
        coords_w = jt.arange(0, w * stride, stride, dtype='float32') + stride / 2
        coords_h = jt.arange(0, h * stride, stride, dtype='float32') + stride / 2
        coords_y, coords_x = jt.meshgrid(coords_h, coords_w)
        coords = jt.stack([coords_x.flatten(), coords_y.flatten()], dim=1)
        return coords

    def _calculate_ltrb_offsets(self, coords, gt_boxes):
        x, y = coords[:, 0], coords[:, 1]
        l_off = x.unsqueeze(0).unsqueeze(2) - gt_boxes[:, :, 0].unsqueeze(1)
        t_off = y.unsqueeze(0).unsqueeze(2) - gt_boxes[:, :, 1].unsqueeze(1)
        r_off = gt_boxes[:, :, 2].unsqueeze(1) - x.unsqueeze(0).unsqueeze(2)
        b_off = gt_boxes[:, :, 3].unsqueeze(1) - y.unsqueeze(0).unsqueeze(2)
        return jt.stack([l_off, t_off, r_off, b_off], dim=3)

    def _get_positive_mask(self, ltrb_off, stride, limit_range, gt_boxes, coords, sample_radiu_ratio=2.5): #should be 1.5
        off_min = jt.min(ltrb_off, dim=3)
        off_max = jt.max(ltrb_off, dim=3)
        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        x, y = coords[:, 0], coords[:, 1]
        c_l_off = x.unsqueeze(0).unsqueeze(2) - gt_center_x.unsqueeze(1)
        c_t_off = y.unsqueeze(0).unsqueeze(2) - gt_center_y.unsqueeze(1)
        c_r_off = gt_center_x.unsqueeze(1) - x.unsqueeze(0).unsqueeze(2)
        c_b_off = gt_center_y.unsqueeze(1) - y.unsqueeze(0).unsqueeze(2)
        c_ltrb_off = jt.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=3)
        c_off_max = jt.max(c_ltrb_off, dim=3)
        mask_center = c_off_max < radiu

        # --- START: ADD THIS DEBUG BLOCK ---
        if gt_boxes.shape[0] > 0 and gt_boxes[0,0,0].item() != -1:
            print(f"  [GenTargets Stride {stride}] Candidates: "
                f"in_box={mask_in_gtboxes.sum().item()}, "
                f"in_level={mask_in_level.sum().item()}, "
                f"in_center={mask_center.sum().item()} -> "
                f"Final={ (mask_in_gtboxes & mask_in_level & mask_center).sum().item() }")
        # --- END: ADD THIS DEBUG BLOCK ---

        return mask_in_gtboxes & mask_in_level & mask_center

    def _get_positive_mask_with_atss(self, ltrb_off, stride, limit_range, gt_boxes, coords, sample_radiu_ratio=1.5):
        # The new ATSS-based positive sample selection logic
        
        # 1. The location must be inside a ground truth box. This is a basic prerequisite.
        off_min = jt.min(ltrb_off, dim=3)
        mask_in_gtboxes = off_min > 0
        
        # 2. Reconstruct the predicted boxes for IoU calculation
        #    coords shape: [num_points, 2] -> unsqueeze to [1, num_points, 1, 2]
        #    ltrb_off shape: [B, num_points, num_gt, 4]
        coords_unsqueezed = coords.unsqueeze(0).unsqueeze(2)
        pred_boxes_x1y1 = coords_unsqueezed - ltrb_off[..., :2]
        pred_boxes_x2y2 = coords_unsqueezed + ltrb_off[..., 2:]
        pred_boxes = jt.concat([pred_boxes_x1y1, pred_boxes_x2y2], dim=-1)

        # 3. Calculate IoU between each GT box and all locations on this level
        ious_list = []
        for b in range(gt_boxes.shape[0]):
            # Filter out padded GT boxes
            gt_boxes_b = gt_boxes[b]
            valid_gt_mask = gt_boxes_b[:, 0] != -1
            gt_boxes_b_valid = gt_boxes_b[valid_gt_mask]
            
            if gt_boxes_b_valid.numel() == 0:
                ious_list.append(jt.zeros((coords.shape[0], gt_boxes.shape[1])))
                continue

            pred_boxes_b_valid = pred_boxes[b][:, valid_gt_mask, :]
            
            # Reshape for efficient IoU calculation and get the diagonal
            # This matches each GT box with the predictions generated for it
            iou_matrix = compute_iou(
                pred_boxes_b_valid.reshape(-1, 4), 
                gt_boxes_b_valid
            ).reshape(coords.shape[0], gt_boxes_b_valid.shape[0], gt_boxes_b_valid.shape[0])
            ious_b_valid = jt.diagonal(iou_matrix, dim1=1, dim2=2)
            
            # Pad back to original GT dimension size
            padded_ious = jt.zeros((coords.shape[0], gt_boxes.shape[1]))
            padded_ious[:, valid_gt_mask] = ious_b_valid
            ious_list.append(padded_ious)
        
        ious = jt.stack(ious_list, dim=0)

        # 4. Calculate the ATSS dynamic threshold (mean + std) for each GT
        #    Use mask_in_gtboxes to ensure stats are calculated only on valid locations
        ious_valid = jt.where(mask_in_gtboxes, ious, jt.zeros_like(ious))
        num_in_gt = mask_in_gtboxes.sum(dim=1, keepdims=True).clamp(min_v=1)
        
        iou_mean = ious_valid.sum(dim=1, keepdims=True) / num_in_gt
        iou_std = jt.sqrt(( (ious_valid - iou_mean).pow(2) * mask_in_gtboxes ).sum(dim=1, keepdims=True) / num_in_gt)
        atss_threshold = iou_mean + iou_std

        # 5. Create the final ATSS mask and combine it with the prerequisite mask
        mask_atss = ious >= atss_threshold
        
        # The final positive mask is the intersection of the two conditions
        final_mask = mask_in_gtboxes & mask_atss

        # --- Debugging Print ---
        if gt_boxes.shape[0] > 0 and gt_boxes[0,0,0].item() != -1:
            print(f"  [GenTargets Stride {stride}] Candidates: "
                f"in_box={mask_in_gtboxes.sum().item()}, "
                f"pass_atss={(mask_atss & mask_in_gtboxes).sum().item()} -> "
                f"Final Positives={final_mask.sum().item()}")

        return final_mask
        
    def _resolve_ambiguity(self, mask_pos, ltrb_off):
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])
        areas = jt.where(mask_pos, areas, jt.ones_like(areas) * 99999999.0)
        areas_min_ind, _ = jt.argmin(areas, dim=2)
        mask_pos_sum = mask_pos.sum(dim=2) >= 1
        return areas_min_ind, mask_pos_sum

    def _assign_targets(self, mask_pos_sum, areas_min_ind, ltrb_off, classes, batch_size, num_points):
        reg_targets = jt.full((batch_size, num_points, 4), -1.0)
        cls_targets = jt.zeros((batch_size, num_points, 1)).int64()
        cnt_targets = jt.full((batch_size, num_points, 1), -1.0)

        if mask_pos_sum.sum().item() > 0:
            # 1. Find the batch and point indices of all positive locations
            b_idx, n_idx = jt.where(mask_pos_sum)
            
            # 2. For these positive locations, get the index of the best GT box
            box_idx = areas_min_ind[b_idx, n_idx]
            
            # 3. Gather the correct ltrb offsets and classes using these indices
            pos_reg_targets = ltrb_off[b_idx, n_idx, box_idx]
            pos_cls_targets = classes[b_idx, box_idx]

            # 4. Assign the gathered values back to the main target tensors
            # Jittor requires a loop for this type of sparse assignment
            for i in range(len(b_idx)):
                b, n = b_idx[i], n_idx[i]
                reg_targets[b, n, :] = pos_reg_targets[i]
                cls_targets[b, n, 0] = pos_cls_targets[i]

            # 5. Calculate and assign centerness targets
            l_ = pos_reg_targets[..., 0].clamp(min_v=0)
            t_ = pos_reg_targets[..., 1].clamp(min_v=0)
            r_ = pos_reg_targets[..., 2].clamp(min_v=0)
            b_ = pos_reg_targets[..., 3].clamp(min_v=0)
            
            left_right_min = jt.minimum(l_, r_)
            left_right_max = jt.maximum(l_, r_)
            top_bottom_min = jt.minimum(t_, b_)
            top_bottom_max = jt.maximum(t_, b_)
            
            pos_cnt_targets = jt.sqrt((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10))

            for i in range(len(b_idx)):
                b, n = b_idx[i], n_idx[i]
                cnt_targets[b, n, 0] = pos_cnt_targets[i]
                
        return cls_targets, cnt_targets, reg_targets

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range):
        batch_size, _, h, w = out[0].shape
        coords = self._generate_level_coords(h, w, stride)
        ltrb_off = self._calculate_ltrb_offsets(coords, gt_boxes)
        mask_pos = self._get_positive_mask(ltrb_off, stride, limit_range, gt_boxes, coords)
        areas_min_ind, mask_pos_sum = self._resolve_ambiguity(mask_pos, ltrb_off)
        cls_targets, cnt_targets, reg_targets = self._assign_targets(
            mask_pos_sum, areas_min_ind, ltrb_off, classes, batch_size, coords.shape[0]
        )
        return (cls_targets, cnt_targets, reg_targets), coords

    def execute(self, inputs):
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes, classes = inputs[1], inputs[2]
        cls_targets_all, cnt_targets_all, reg_targets_all, coords_all = [], [], [], []
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets, level_coords = self._gen_level_targets(
                level_out, gt_boxes, classes, self.strides[level], self.limit_range[level]
            )
            cls_targets_all.append(level_targets[0])
            cnt_targets_all.append(level_targets[1])
            reg_targets_all.append(level_targets[2])
            coords_all.append(level_coords)
        return (
            jt.concat(cls_targets_all, dim=1),
            jt.concat(cnt_targets_all, dim=1),
            jt.concat(reg_targets_all, dim=1),
            jt.concat(coords_all, dim=0)
        )

# --- Compute Loss Wrapper Functions ---
def compute_cls_loss(preds, targets, mask, class_num=80):
    B = preds[0].shape[0]
    preds_reshape = []
    for p in preds:
        preds_reshape.append(p.permute(0, 2, 3, 1).reshape(B, -1, class_num))
    preds = jt.concat(preds_reshape, dim=1).reshape(-1, class_num)
    mask = mask.reshape(-1)
    targets = targets.reshape(-1, 1)
    num_pos = mask.sum().clamp(min_v=1.0)
    preds_pos = preds[mask]
    targets_pos = targets[mask]
    targets_one_hot = jt.zeros_like(preds_pos)
    targets_one_hot = targets_one_hot.scatter(1, targets_pos.int() - 1, jt.array(1.0))
    loss = focal_loss_from_logits(preds_pos, targets_one_hot)
    return loss / num_pos

def compute_cnt_loss(preds, targets, mask):
    B = preds[0].shape[0]
    preds_reshape = []
    for p in preds:
        preds_reshape.append(p.permute(0, 2, 3, 1).reshape(B, -1, 1))
    preds = jt.concat(preds_reshape, dim=1).reshape(-1)
    mask = mask.reshape(-1)
    targets = targets.reshape(-1)
    num_pos = mask.sum().clamp(min_v=1.0)
    preds_pos = preds[mask]
    targets_pos = targets[mask]
    loss = jt.nn.binary_cross_entropy_with_logits(preds_pos, targets_pos, size_average=False)
    return loss / num_pos

def compute_reg_loss(preds, targets, mask, coords):
    B = preds[0].shape[0]
    preds_reshape = []
    for p in preds:
        preds_reshape.append(p.permute(0, 2, 3, 1).reshape(B, -1, 4))
    preds = jt.concat(preds_reshape, dim=1)
    
    num_pos = mask.sum().clamp(min_v=1.0)
    
    mask_b = mask.reshape(B, -1)
    preds_pos = preds[mask_b]
    targets_pos = targets.reshape(B, -1, 4)[mask_b]
    coords_pos = coords.unsqueeze(0).expand(B, -1, -1)[mask_b]


    # --- START: BOX RECONSTRUCTION DEBUG BLOCK ---
    if preds_pos.numel() > 0:
        print("\n    --- BOX RECONSTRUCTION DEBUG ---")
        pred_boxes_x1y1 = coords_pos - preds_pos[:, :2]
        pred_boxes_x2y2 = coords_pos + preds_pos[:, 2:]
        pred_boxes = jt.concat([pred_boxes_x1y1, pred_boxes_x2y2], dim=1)
        
        target_boxes_x1y1 = coords_pos - targets_pos[:, :2]
        target_boxes_x2y2 = coords_pos + targets_pos[:, 2:]
        target_boxes = jt.concat([target_boxes_x1y1, target_boxes_x2y2], dim=1)
        
        print(f"    [BOX_RECON] Feature Point Coords (sample):\n{coords_pos[:2].numpy()}")
        print(f"    [BOX_RECON] Predicted LTRB Offsets (sample):\n{preds_pos[:2].numpy()}")
        print(f"    [BOX_RECON] Reconstructed PREDICTED Box (x1y1x2y2):\n{pred_boxes[:2].numpy()}")
        print(f"    [BOX_RECON] Target LTRB Offsets (sample):\n{targets_pos[:2].numpy()}")
        print(f"    [BOX_RECON] Reconstructed TARGET Box (x1y1x2y2):\n{target_boxes[:2].numpy()}")
    else: # If there are no positive samples, create empty tensors
        pred_boxes = jt.empty((0, 4))
        target_boxes = jt.empty((0, 4))
    # --- END: BOX RECONSTRUCTION DEBUG BLOCK ---

    
    #pred_boxes_x1y1 = coords_pos - preds_pos[:, :2]
    #pred_boxes_x2y2 = coords_pos + preds_pos[:, 2:]
    #pred_boxes = jt.concat([pred_boxes_x1y1, pred_boxes_x2y2], dim=1)
    
    #target_boxes_x1y1 = coords_pos - targets_pos[:, :2]
    #target_boxes_x2y2 = coords_pos + targets_pos[:, 2:]
    #target_boxes = jt.concat([target_boxes_x1y1, target_boxes_x2y2], dim=1)
    
    loss = giou_loss(pred_boxes, target_boxes)
    return loss / num_pos

# --- Main LOSS Class ---
class LOSS(nn.Module):
    def __init__(self, config=None):
        super(LOSS, self).__init__()
        if config is None:
            class TempConfig:
                add_centerness = True
            self.config = TempConfig()
        else:
            self.config = config

    def execute(self, preds, targets_with_coords):
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets, coords = targets_with_coords
        
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)
        
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos)
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask_pos)
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos, coords)
        
        if self.config.add_centerness:
            total_loss = cls_loss + cnt_loss + reg_loss
        else:
            total_loss = cls_loss + reg_loss
            
        return cls_loss, cnt_loss, reg_loss, total_loss




#--test--







if __name__ == '__main__':
    import torch
    import torch.nn as nn_torch
    import torch.nn.functional as F
    import numpy as np

    jt.flags.use_cuda = 1
    np.random.seed(42) # Use a fixed seed for reproducible tests

    # --- Test Fixtures: Create realistic mock data ---
    B, C, R = 2, 80, 4
    strides = [8, 16, 32]
    limit_range = [[-1, 64], [64, 128], [128, 999999]]
    fpn_dims = [(20, 20), (10, 10), (5, 5)]

    # Mock ground truth
    gt_boxes_np = np.array([
        [[10, 20, 50, 60.5], [70, 80, 100, 110]], # Image 1: 2 boxes
        [[30, 30, 90, 95.2], [1, 1, 150, 150]]    # Image 2: 2 boxes
    ]).astype('float32')
    classes_np = np.array([[1, 5], [3, 8]]).astype('int64')

    # Mock predictions from the model's head
    cls_preds_np = [np.random.randn(B, C, h, w).astype('float32') for h, w in fpn_dims]
    cnt_preds_np = [np.random.randn(B, 1, h, w).astype('float32') for h, w in fpn_dims]
    reg_preds_np = [np.random.randn(B, R, h, w).astype('float32') for h, w in fpn_dims]

    # ===================================================================
    # START: Self-Contained PyTorch Reference Implementation
    # ===================================================================
    class PyTorchDefaultConfig:
        add_centerness = True

    def focal_loss_from_logits_pt(preds, targets, gamma=2.0, alpha=0.25):
        preds = preds.sigmoid()
        pt = preds * targets + (1.0 - preds) * (1.0 - targets)
        w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
        return loss.sum()

    def giou_loss_pt(preds, targets):
        px1, py1, px2, py2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        tx1, ty1, tx2, ty2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]
        pred_areas = (px2 - px1) * (py2 - py1)
        target_areas = (tx2 - tx1) * (ty2 - ty1)
        ix1, iy1, ix2, iy2 = torch.max(px1, tx1), torch.max(py1, ty1), torch.min(px2, tx2), torch.min(py2, ty2)
        inter_w, inter_h = (ix2 - ix1).clamp(min=0), (iy2 - iy1).clamp(min=0)
        overlap = inter_w * inter_h
        union = pred_areas + target_areas - overlap
        iou = overlap / union.clamp(1e-10)
        ex1, ey1, ex2, ey2 = torch.min(px1, tx1), torch.min(py1, ty1), torch.max(px2, tx2), torch.max(py2, ty2)
        G_area = (ex2 - ex1) * (ey2 - ey1)
        giou = iou - (G_area - union) / G_area.clamp(1e-10)
        loss = 1.0 - giou
        return loss.sum()
    
    class GenTargets_pt(nn_torch.Module):
        def __init__(self, strides, limit_range):
            super().__init__()
            self.strides = strides
            self.limit_range = limit_range
        def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
            batch_size, _, h, w = out[0].shape
            device = out[0].device
            shifts_x = torch.arange(0, w*stride, stride, dtype=torch.float32, device=device) + stride/2
            shifts_y = torch.arange(0, h*stride, stride, dtype=torch.float32, device=device) + stride/2
            shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing='xy')
            coords = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], -1)
            x, y = coords[:,0], coords[:,1]
            l_off = x[None,:,None] - gt_boxes[...,0][:,None,:]
            t_off = y[None,:,None] - gt_boxes[...,1][:,None,:]
            r_off = gt_boxes[...,2][:,None,:] - x[None,:,None]
            b_off = gt_boxes[...,3][:,None,:] - y[None,:,None]
            ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)
            areas = (ltrb_off[...,0]+ltrb_off[...,2]) * (ltrb_off[...,1]+ltrb_off[...,3])
            off_min, off_max = torch.min(ltrb_off, dim=-1).values, torch.max(ltrb_off, dim=-1).values
            mask_in_gtboxes = off_min > 0
            mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
            radiu = stride * sample_radiu_ratio
            gt_center_x, gt_center_y = (gt_boxes[...,0]+gt_boxes[...,2])/2, (gt_boxes[...,1]+gt_boxes[...,3])/2
            c_l_off = x[None,:,None] - gt_center_x[:,None,:]; c_t_off = y[None,:,None] - gt_center_y[:,None,:]
            c_r_off = gt_center_x[:,None,:] - x[None,:,None]; c_b_off = gt_center_y[:,None,:] - y[None,:,None]
            c_ltrb_off = torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)
            c_off_max = torch.max(c_ltrb_off,dim=-1).values
            mask_center = c_off_max < radiu
            mask_pos = mask_in_gtboxes & mask_in_level & mask_center
            areas[~mask_pos] = 99999999
            areas_min_ind = torch.min(areas, dim=-1).indices
            reg_targets = torch.full((batch_size, coords.shape[0], 4), -1.0, device=device)
            cls_targets = torch.zeros((batch_size, coords.shape[0], 1), dtype=torch.long, device=device)
            cnt_targets = torch.full((batch_size, coords.shape[0], 1), -1.0, device=device)
            mask_pos_2 = mask_pos.long().sum(dim=-1) >= 1
            if mask_pos_2.sum() > 0:
                pos_reg_targets = ltrb_off[mask_pos_2, areas_min_ind[mask_pos_2]]
                reg_targets[mask_pos_2] = pos_reg_targets
                pos_cls_targets = classes.unsqueeze(1).repeat(1,coords.shape[0],1)[mask_pos_2,areas_min_ind[mask_pos_2]]
                cls_targets[mask_pos_2] = pos_cls_targets.unsqueeze(1)
                l_,t_,r_,b_ = pos_reg_targets.clamp(min=0).chunk(4,dim=-1)
                # CORRECTED: Removed the erroneous .unsqueeze(dim=-1)
                pos_cnt_targets = torch.sqrt((torch.min(l_,r_)*torch.min(t_,b_))/(torch.max(l_,r_)*torch.max(t_,b_)+1e-10))
                cnt_targets[mask_pos_2] = pos_cnt_targets
            return (cls_targets, cnt_targets, reg_targets), coords
        def forward(self, inputs):
            preds, gt_boxes, classes = inputs
            cls_logits, _, _ = preds
            cls_targets_all, cnt_targets_all, reg_targets_all ,coords_all = [],[],[],[]
            for i in range(len(self.strides)):
                targets, coords = self._gen_level_targets([cls_logits[i]], gt_boxes, classes, self.strides[i], self.limit_range[i])
                cls_targets_all.append(targets[0]); cnt_targets_all.append(targets[1]); reg_targets_all.append(targets[2]); coords_all.append(coords)
            return (torch.cat(cls_targets_all,dim=1), torch.cat(cnt_targets_all,dim=1), torch.cat(reg_targets_all,dim=1), torch.cat(coords_all,dim=0))

    def compute_cls_loss_pt(preds, targets, mask, class_num=80):
        B = preds[0].shape[0]
        preds_r = [p.permute(0,2,3,1).reshape(B,-1,class_num) for p in preds]
        preds = torch.cat(preds_r,dim=1).reshape(-1,class_num)
        mask, targets = mask.reshape(-1), targets.reshape(-1,1)
        num_pos = mask.sum().clamp(min=1.0)
        preds_pos = preds[mask]; targets_pos = targets[mask]
        targets_one_hot = torch.zeros_like(preds_pos).scatter_(1,targets_pos.long()-1,1.0)
        return focal_loss_from_logits_pt(preds_pos, targets_one_hot)/num_pos

    def compute_cnt_loss_pt(preds, targets, mask):
        B = preds[0].shape[0]
        preds_r = [p.permute(0,2,3,1).reshape(B,-1,1) for p in preds]
        preds = torch.cat(preds_r,dim=1).reshape(-1)
        mask, targets = mask.reshape(-1), targets.reshape(-1)
        num_pos = mask.sum().clamp(min=1.0)
        preds_pos, targets_pos = preds[mask], targets[mask]
        return F.binary_cross_entropy_with_logits(preds_pos, targets_pos, reduction='sum')/num_pos
    
    def compute_reg_loss_pt(preds, targets, mask, coords):
        B = preds[0].shape[0]
        preds_r = [p.permute(0,2,3,1).reshape(B,-1,4) for p in preds]
        preds = torch.cat(preds_r,dim=1)
        mask_b = mask.reshape(B,-1)
        preds_pos = preds[mask_b]
        targets_pos = targets.reshape(B,-1,4)[mask_b]
        coords_pos = coords.unsqueeze(0).expand(B,-1,-1)[mask_b]
        pred_boxes = torch.cat([coords_pos-preds_pos[:,:2], coords_pos+preds_pos[:,2:]], dim=1)
        target_boxes = torch.cat([coords_pos-targets_pos[:,:2], coords_pos+targets_pos[:,2:]], dim=1)
        return giou_loss_pt(pred_boxes, target_boxes)/mask.sum().clamp(min=1.0)

    class LOSS_pt(nn_torch.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = PyTorchDefaultConfig() if config is None else config
        def forward(self, inputs):
            preds, targets, coords = inputs
            cls_logits, cnt_logits, reg_preds = preds
            cls_targets, cnt_targets, reg_targets = targets
            mask_pos = (cnt_targets > -1).squeeze(dim=-1)
            cls_loss = compute_cls_loss_pt(cls_logits,cls_targets,mask_pos)
            cnt_loss = compute_cnt_loss_pt(cnt_logits,cnt_targets,mask_pos)
            reg_loss = compute_reg_loss_pt(reg_preds,reg_targets,mask_pos,coords)
            total_loss = cls_loss + cnt_loss + reg_loss if self.config.add_centerness else cls_loss + reg_loss
            return cls_loss, cnt_loss, reg_loss, total_loss

    # ===================================================================
    # END: PyTorch Reference Implementation
    # ===================================================================

    # --- Jittor Full Pipeline ---
    print("--- Running Full Jittor Pipeline ---")
    target_generator_jt = GenTargets(strides, limit_range)
    loss_computer_jt = LOSS()
    preds_jt_list = [[jt.array(p) for p in arr] for arr in [cls_preds_np, cnt_preds_np, reg_preds_np]]
    gt_boxes_jt, classes_jt = jt.array(gt_boxes_np), jt.array(classes_np)
    targets_jt_tuple = target_generator_jt([preds_jt_list, gt_boxes_jt, classes_jt])
    cls_loss_jt, cnt_loss_jt, reg_loss_jt, total_loss_jt = loss_computer_jt(preds_jt_list, targets_jt_tuple)
    print(f"[Jittor]   Cls: {cls_loss_jt.item():.4f}, Cnt: {cnt_loss_jt.item():.4f}, Reg: {reg_loss_jt.item():.4f}, Total: {total_loss_jt.item():.4f}")

    # --- PyTorch Full Pipeline ---
    print("\n--- Running Full PyTorch Pipeline ---")
    device = torch.device("cuda" if jt.flags.use_cuda else "cpu")
    target_generator_pt = GenTargets_pt(strides, limit_range)
    loss_computer_pt = LOSS_pt()
    preds_pt_list = [[torch.from_numpy(p).to(device) for p in arr] for arr in [cls_preds_np, cnt_preds_np, reg_preds_np]]
    gt_boxes_pt, classes_pt = torch.from_numpy(gt_boxes_np).to(device), torch.from_numpy(classes_np).to(device)
    
    # CORRECTED: Pass the full prediction list to the reference GenTargets
    targets_pt_tuple = target_generator_pt([preds_pt_list, gt_boxes_pt, classes_pt])
    
    targets_pt_list = targets_pt_tuple[:-1]
    coords_pt = targets_pt_tuple[-1]
    cls_loss_pt, cnt_loss_pt, reg_loss_pt, total_loss_pt = loss_computer_pt([preds_pt_list, targets_pt_list, coords_pt])
    print(f"[PyTorch]  Cls: {cls_loss_pt.item():.4f}, Cnt: {cnt_loss_pt.item():.4f}, Reg: {reg_loss_pt.item():.4f}, Total: {total_loss_pt.item():.4f}")

    # --- Final Comparison ---
    print("\n" + "="*40)
    print("--- Final End-to-End Comparison ---")
    assert np.allclose(cls_loss_jt.item(), cls_loss_pt.item(), atol=1e-4), "Final Classification Loss MISMATCH!"
    print("✅ Classification Loss Matches")
    assert np.allclose(cnt_loss_jt.item(), cnt_loss_pt.item(), atol=1e-4), "Final Centerness Loss MISMATCH!"
    print("✅ Centerness Loss Matches")
    assert np.allclose(reg_loss_jt.item(), reg_loss_pt.item(), atol=1e-4), "Final Regression Loss MISMATCH!"
    print("✅ Regression Loss Matches")
    print("\n🎉🎉🎉 Congratulations! The entire loss.py file has been successfully translated and verified!")






















