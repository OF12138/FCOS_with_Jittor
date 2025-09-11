#
# loss.py (PyTorch Version)
# This file uses the PyTorch framework for all loss-related calculations.
# It includes helper functions to convert Jittor tensors to PyTorch tensors at the beginning
# of the computation and convert the final PyTorch loss tensor back to a Jittor tensor at the end.
# This allows it to function as a drop-in replacement in your Jittor project.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import jittor as jt # Jittor is imported ONLY for the final conversion back.
from .config import DefaultConfig

def coords_fmap2orig(feature, stride):
    """
    Convert feature map coordinates to image coordinates (Jittor version).
    This function is used by DetectHead in FCOS.py during inference.
    """
    h, w = feature.shape[-2:]
    shifts_x = jt.arange(0, w, dtype=jt.float32) + 0.5
    shifts_y = jt.arange(0, h, dtype=jt.float32) + 0.5
    shifts_x = shifts_x * stride
    shifts_y = shifts_y * stride
    shift_y, shift_x = jt.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])
    coords = jt.stack([shift_x, shift_y], dim=-1)
    return coords

# -------------------- Jittor <--> PyTorch Conversion Helpers --------------------

def jittor_to_torch(jittor_var):
    """Recursively converts a Jittor Var or a list of Jittor Vars to PyTorch Tensors."""
    if jittor_var is None:
        return None
    if isinstance(jittor_var, list):
        return [jittor_to_torch(v) for v in jittor_var]
    
    # Convert Jittor Var to numpy array, then to PyTorch tensor.
    # Assumes CUDA is used, as per your train.py (jt.flags.use_cuda=1)
    return torch.from_numpy(jittor_var.numpy()).cuda()

def torch_to_jittor(torch_tensor):
    """Converts a PyTorch Tensor back to a Jittor Var."""
    # Convert PyTorch tensor to numpy array (after moving to CPU), then to Jittor Var.
    return jt.array(torch_tensor.detach().cpu().numpy())

# -------------------- Target generation (PyTorch) --------------------
class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(self.strides) == len(self.limit_range)

    def forward(self, inputs): # RENAMED from execute to forward
        # 1. CONVERT JITTOR INPUTS TO PYTORCH
        cls_logits, cnt_logits, reg_preds = jittor_to_torch(inputs[0])
        gt_boxes, classes, batch_scales = jittor_to_torch(inputs[1]), jittor_to_torch(inputs[2]), jittor_to_torch(inputs[3])
        
        # --- All internal calculations are now in PyTorch ---
        cls_targets_all, cnt_targets_all, reg_targets_all = [], [], []

        for lvl, stride in enumerate(self.strides):
            # Pass feature map from current level for shape info
            cls_t, cnt_t, reg_t = self._gen_level_targets(
                cls_logits[lvl], gt_boxes, classes,
                stride, self.limit_range[lvl]
            )
            cls_targets_all.append(cls_t)
            cnt_targets_all.append(cnt_t)
            reg_targets_all.append(reg_t)

        # 2. CONVERT PYTORCH RESULTS BACK TO JITTOR
        cls_targets_final = torch_to_jittor(torch.cat(cls_targets_all, dim=1))
        cnt_targets_final = torch_to_jittor(torch.cat(cnt_targets_all, dim=1))
        reg_targets_final = torch_to_jittor(torch.cat(reg_targets_all, dim=1))

        return [cls_targets_final, cnt_targets_final, reg_targets_final]

    def _gen_level_targets(self, feature_map, gt_boxes, classes, stride, limit_range, sample_radius_ratio=1.5):
        B, _, H, W = feature_map.shape
        HW = H * W
        min_range, max_range = float(limit_range[0]), float(limit_range[1])

        shifts_x = torch.arange(0, W * stride, stride, dtype=torch.float32, device=feature_map.device) + stride / 2.0
        shifts_y = torch.arange(0, H * stride, stride, dtype=torch.float32, device=feature_map.device) + stride / 2.0
        # Use 'xy' indexing for clarity, matching standard convention
        shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing='xy')
        coords_original = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1)

        cls_list, cnt_list, reg_list = [], [], []
        INF = 1e10

        for b in range(B):
            boxes_b, cls_b = gt_boxes[b], classes[b]
            valid_mask = (cls_b >= 0)
            
            boxes_valid = boxes_b[valid_mask]
            cls_valid = cls_b[valid_mask].long()
            m = boxes_valid.shape[0]

            if m == 0:
                cls_list.append(torch.zeros((HW, 1), dtype=torch.int64, device=feature_map.device))
                cnt_list.append(torch.full((HW, 1), -1.0, dtype=torch.float32, device=feature_map.device))
                reg_list.append(torch.full((HW, 4), -1.0, dtype=torch.float32, device=feature_map.device))
                continue

            coords_b = coords_original
            coords_exp = coords_b.unsqueeze(1).repeat(1, m, 1)
            boxes_exp = boxes_valid.unsqueeze(0).repeat(HW, 1, 1)

            x, y = coords_exp[..., 0], coords_exp[..., 1]
            l = x - boxes_exp[..., 0]
            t = y - boxes_exp[..., 1]
            r = boxes_exp[..., 2] - x
            b_ = boxes_exp[..., 3] - y
            ltrb = torch.stack([l, t, r, b_], dim=-1)

            off_min = torch.min(ltrb, dim=-1).values
            off_max = torch.max(ltrb, dim=-1).values
            mask_in_box = off_min >= 0
            mask_in_level = (off_max >= min_range) & (off_max <= max_range)

            cx = (boxes_valid[:, 0] + boxes_valid[:, 2]) / 2.0
            cy = (boxes_valid[:, 1] + boxes_valid[:, 3]) / 2.0
            c_l = x - cx.unsqueeze(0)
            c_t = y - cy.unsqueeze(0)
            c_r = cx.unsqueeze(0) - x
            c_b = cy.unsqueeze(0) - y
            c_off_max = torch.max(torch.stack([c_l, c_t, c_r, c_b], dim=-1), dim=-1).values
            radius = stride * sample_radius_ratio
            mask_center = c_off_max < radius

            pos_mask = mask_in_box & mask_in_level & mask_center

            areas = (boxes_valid[:, 2] - boxes_valid[:, 0]) * (boxes_valid[:, 3] - boxes_valid[:, 1])
            areas_exp = areas.unsqueeze(0).repeat(HW, 1)
            areas_valid = torch.where(pos_mask, areas_exp, torch.full_like(areas_exp, INF))

            min_inds = torch.argmin(areas_valid, dim=1)
            min_vals = areas_valid.gather(1, min_inds.unsqueeze(1)).squeeze(1)

            cls_target = torch.zeros((HW, 1), dtype=torch.int64, device=feature_map.device)
            cnt_target = torch.full((HW, 1), -1.0, dtype=torch.float32, device=feature_map.device)
            reg_target = torch.full((HW, 4), -1.0, dtype=torch.float32, device=feature_map.device)

            pos_loc_mask = min_vals < INF
            
            if pos_loc_mask.sum() > 0:
                pos_indices = torch.where(pos_loc_mask)[0]
                min_inds_pos = min_inds[pos_indices]

                reg_selected = ltrb[pos_indices, min_inds_pos]
                cls_selected = cls_valid[min_inds_pos]

                lr_min = torch.min(reg_selected[:, 0], reg_selected[:, 2])
                lr_max = torch.max(reg_selected[:, 0], reg_selected[:, 2]).clamp(min=1e-5)
                tb_min = torch.min(reg_selected[:, 1], reg_selected[:, 3])
                tb_max = torch.max(reg_selected[:, 1], reg_selected[:, 3]).clamp(min=1e-5)
                centerness = torch.sqrt((lr_min * tb_min) / (lr_max * tb_max + 1e-10))
                
                cls_target[pos_indices] = cls_selected.unsqueeze(1)
                reg_target[pos_indices] = reg_selected
                cnt_target[pos_indices] = centerness.unsqueeze(1)

            cls_list.append(cls_target)
            cnt_list.append(cnt_target)
            reg_list.append(reg_target)

        return torch.stack(cls_list, 0), torch.stack(cnt_list, 0), torch.stack(reg_list, 0)


# -------------------- Loss functions (PyTorch) --------------------
def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    preds = preds.sigmoid()
    pt = preds * targets + (1 - preds) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return (-w * (1 - pt).pow(gamma) * torch.log(pt + 1e-12)).sum()

def l1_loss(preds, targets):
    return F.l1_loss(preds, targets, reduction='sum')

def compute_cls_loss(preds, targets, mask, class_num=81):
    batch_size = targets.shape[0]
    preds_reshape = [p.permute(0, 2, 3, 1).reshape(batch_size, -1, class_num) for p in preds]
    preds = torch.cat(preds_reshape, dim=1)
    
    mask = mask.unsqueeze(-1)
    num_pos = torch.sum(mask.float()).clamp(min=1.0)
    
    # Check for empty mask before indexing
    if num_pos == 0:
        return torch.tensor(0.0, device=preds.device)
        
    preds_pos = preds[mask.expand_as(preds)].reshape(-1, class_num)
    targets_pos = targets[mask.squeeze(-1)].reshape(-1, 1) # Corrected mask indexing for targets

    targets_one_hot = torch.zeros_like(preds_pos).scatter_(1, targets_pos.long() - 1, 1) # COCO classes are 1-80
    
    loss = focal_loss_from_logits(preds_pos, targets_one_hot)
    return loss / num_pos

def compute_cnt_loss(preds, targets, mask):
    preds_reshape = [p.permute(0, 2, 3, 1).reshape(targets.shape[0], -1, 1) for p in preds]
    preds = torch.cat(preds_reshape, dim=1)
    
    mask = mask.unsqueeze(-1)
    num_pos = torch.sum(mask.float()).clamp(min=1.0)
    
    if num_pos == 0:
        return torch.tensor(0.0, device=preds.device)
        
    preds_pos = preds[mask]
    targets_pos = targets[mask]

    loss = F.binary_cross_entropy_with_logits(preds_pos, targets_pos, reduction='sum')
    return loss / num_pos

def compute_reg_loss(preds, targets, mask):
    preds_reshape = [p.permute(0, 2, 3, 1).reshape(targets.shape[0], -1, 4) for p in preds]
    preds = torch.cat(preds_reshape, dim=1)
    
    mask = mask.unsqueeze(-1)
    num_pos = torch.sum(mask.float()).clamp(min=1.0)
    
    if num_pos == 0:
        return torch.tensor(0.0, device=preds.device)

    preds_pos = preds[mask.expand_as(preds)].reshape(-1, 4)
    targets_pos = targets[mask.expand_as(targets)].reshape(-1, 4)

    # L1 loss is more stable and common for regressing l,t,r,b distances directly
    loss = l1_loss(preds_pos, targets_pos)
    return loss / num_pos

# -------------------- Loss wrapper (PyTorch) --------------------
class LOSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = DefaultConfig if config is None else config
    
    def forward(self, inputs): # RENAMED from execute to forward
        # 1. CONVERT JITTOR INPUTS TO PYTORCH
        preds, targets = jittor_to_torch(inputs)
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        
        # --- All internal calculations are now in PyTorch ---
        mask = (cnt_targets > -1)
        if mask.ndim == 3: mask = mask.squeeze(-1)

        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask, self.config.class_num)
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask)
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask)

        if self.config.add_centerness:
            total = cls_loss + cnt_loss + reg_loss
        else:
            total = cls_loss + reg_loss
        
        # 2. CONVERT PYTORCH RESULTS BACK TO JITTOR
        return (torch_to_jittor(cls_loss),
                torch_to_jittor(cnt_loss),
                torch_to_jittor(reg_loss),
                torch_to_jittor(total))