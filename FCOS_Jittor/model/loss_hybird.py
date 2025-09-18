import torch
import torch.nn as nn
import torch.nn.functional as F
import jittor as jt
from .config import DefaultConfig

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

# -------------------- Jittor <--> PyTorch Conversion Helpers (CPU-safe) --------------------
#can't pass the gradients.
def jittor_to_torch(jittor_var):
    if jittor_var is None: return None
    if isinstance(jittor_var, list): return [jittor_to_torch(v) for v in jittor_var]
    return torch.from_numpy(jittor_var.numpy())

def torch_to_jittor(torch_tensor):
    return jt.array(torch_tensor.detach().cpu().numpy())

# -------------------- Target generation (Logic from Pure PyTorch version) --------------------
# this is a fucntion which determine which select the meaningful prediction
class GenTargets(nn.Module):
    '''
    This class implements the FCOS target assignment strategy. 
    For each level of the feature pyramid, 
    it determines which grid cells are inside a ground-truth box, 
    are of an appropriate size for that level, 
    and are close to the center of the box. For these "positive" cells, 
    it generates the classification, centerness, and regression targets.
    '''
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides # Store the stride for each feature map level (e.g., [8, 16, 32, 64, 128])
        self.limit_range = limit_range # Store the size range of objects to be detected at each level.

    def forward(self, inputs):
        #the inputs is [out,batch_boxes,batch_classes,batch_scales]
        #the out is [cls_logits,cnt_logits,reg_preds], which is the output of the FCOS body
        #the batch_boxes is the GT boxes, shape: [batch_size, num_boxes, 4] 
        #the batch_classes is the ground-truth classes
        #the batch_scales is needed to correctly map coordinates.
        # 1. CONVERT JITTOR TO PYTORCH (on CPU)
        cls_logits, cnt_logits, reg_preds = jittor_to_torch(inputs[0])
        gt_boxes, classes, _ = jittor_to_torch(inputs[1]), jittor_to_torch(inputs[2]), jittor_to_torch(inputs[3])
        
        #this block  is to select the device
        device = torch.device("cuda" if jt.flags.use_cuda else "cpu")
        cls_logits = [t.to(device) for t in cls_logits]
        cnt_logits = [t.to(device) for t in cnt_logits]
        reg_preds = [t.to(device) for t in reg_preds]
        gt_boxes, classes = gt_boxes.to(device), classes.to(device)
        
        #
        cls_targets_all, cnt_targets_all, reg_targets_all ,coords_all = [], [], [],[]
        # loop through each FPN level to generate targets for the level
        for level in range(len(cls_logits)): 
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            #call the helper function
            level_targets ,level_coords = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level], self.limit_range[level])
            cls_targets_all.append(level_targets[0])
            cnt_targets_all.append(level_targets[1])
            reg_targets_all.append(level_targets[2])
            coords_all.append(level_coords)
        
        # 2. CONVERT PYTORCH RESULTS BACK TO JITTOR
        #first concatenate (each level to a big tensor ) and then convert
        cls_targets_final = torch_to_jittor(torch.cat(cls_targets_all, dim=1))
        cnt_targets_final = torch_to_jittor(torch.cat(cnt_targets_all, dim=1))
        reg_targets_final = torch_to_jittor(torch.cat(reg_targets_all, dim=1))
        coords_final=torch_to_jittor(torch.cat(coords_all,dim=0))
        return [cls_targets_final, cnt_targets_final, reg_targets_final, coords_final]

    # This is the core logic function
    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        cls_logits, cnt_logits, reg_preds = out
        batch_size, class_num, h, w = cls_logits.shape
        
        # Correctly generate coordinates in (x, y) order
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=cls_logits.device) + stride / 2
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=cls_logits.device) + stride / 2
        shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing='xy')
        coords = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], -1)
        h_mul_w = coords.shape[0]

        x = coords[:, 0]
        y = coords[:, 1]

        #Condition 1 : the gird point must be inside a GT box
        #which means all the 4 distance should be positive
        # calculate the ltrb of each point in the map
        # It seems like the gt_boxes provide the top-left and the bottom-right coordinates
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)

        #the area of the ground truth box
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])

        off_min = torch.min(ltrb_off, dim=-1).values
        off_max = torch.max(ltrb_off, dim=-1).values

        mask_in_gtboxes = off_min > 0

        #Condition 2 the object size must match the level's responsible range
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        #the broadcast
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1).values
        mask_center = c_off_max < radiu

        # A point is a positive candidate if it satisfy all the conditions
        mask_pos = mask_in_gtboxes & mask_in_level & mask_center

        # --- ADD THIS DEBUGGING BLOCK ---
         #This will print the number of potential positive samples found at each stage for each FPN level.
        print(f"  [LOSS-DEBUG] FPN Stride {stride}: "
              f"in_box={mask_in_gtboxes.sum().item()}, "
              f"in_level={mask_in_level.sum().item()}, "
              f"in_center={mask_center.sum().item()} -> "
              f"Final Candidates={mask_pos.sum().item()}")
        # --- END OF DEBUGGING BLOCK ---


        #-----Ambiguity Resolution------
        # If a point is a candidate for multiple GT boxes, assign it to the one with the smallest area.
        areas[~mask_pos] = 99999999 #set the non-positive locations' areas to infinity
        areas_min_ind = torch.min(areas, dim=-1).indices

        #Initialize the target tensors (with non-negatives)
        reg_targets = torch.full((batch_size, h_mul_w, 4), -1.0, device=gt_boxes.device)
        cls_targets = torch.zeros((batch_size, h_mul_w, 1), dtype=torch.long, device=gt_boxes.device)
        cnt_targets = torch.full((batch_size, h_mul_w, 1), -1.0, device=gt_boxes.device)

        mask_pos_2 = mask_pos.long().sum(dim=-1) >= 1

        if mask_pos_2.sum() > 0:
            #For all positive locations, get the ltrb distance to their assigned boxes
            pos_reg_targets = ltrb_off[mask_pos_2, areas_min_ind[mask_pos_2]]
            reg_targets[mask_pos_2] = pos_reg_targets
            
            #get the class lebal for the assigned box
            pos_cls_targets = classes.unsqueeze(1).repeat(1, h_mul_w, 1)[mask_pos_2, areas_min_ind[mask_pos_2]]
            cls_targets[mask_pos_2] = pos_cls_targets.unsqueeze(1)

            l_ = pos_reg_targets[..., 0].clamp(min=0)
            t_ = pos_reg_targets[..., 1].clamp(min=0)
            r_ = pos_reg_targets[..., 2].clamp(min=0)
            b_ = pos_reg_targets[..., 3].clamp(min=0)

            left_right_min = torch.min(l_, r_)
            left_right_max = torch.max(l_, r_)
            top_bottom_min = torch.min(t_, b_)
            top_bottom_max = torch.max(t_, b_)

            # Calculate the centerness Target
            pos_cnt_targets = torch.sqrt((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).unsqueeze(dim=-1)
            cnt_targets[mask_pos_2] = pos_cnt_targets

        # inside target generation, after computing pos_mask or pos_inds
        pos_num_all = mask_pos.sum().item() if hasattr(mask_pos,'sum') else float('nan')
        print(f"DBG_TARGET: pos_total={pos_num_all}")  # if you have per-level masks
        # also show mask_center, mask_in_box
        try:
            print("DBG_TARGET masks:", "mask_in_box", mask_in_gtboxes.sum().item(),
                "mask_center", mask_center.sum().item(), "pos_mask", mask_pos.sum().item())
        except:
            pass


        return (cls_targets, cnt_targets, reg_targets), coords

# -------------------- Loss functions (from Pure PyTorch version) --------------------
def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    '''implementation of focal_loss'''
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    return loss.sum()


def giou_loss(preds, targets):
    # preds and targets are [N, 4] tensors of (l,t,r,b) distances.

    # --- START OF DETAILED DEBUG BLOCK ---
    print("\n    ---------------- INSIDE GIOU_LOSS DEBUG ----------------")
    # Print a sample of the input predictions and targets
    print(f"    [GIOU_DBG] Input Preds (l,t,r,b) sample:\n{preds[:2].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] Input Targets (l,t,r,b) sample:\n{targets[:2].detach().cpu().numpy()}")
    # --- END OF DETAILED DEBUG BLOCK ---

    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (rb_min + lt_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]

    # Corrected area calculation
    area1 = (preds[:, 0] + preds[:, 2]) * (preds[:, 1] + preds[:, 3])
    area2 = (targets[:, 0] + targets[:, 2]) * (targets[:, 1] + targets[:, 3])
    
    union = (area1 + area2 - overlap)
    iou = overlap / union.clamp(1e-10)

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (rb_max + lt_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]
    
    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    loss = 1. - giou
    
    # --- START OF DETAILED DEBUG BLOCK ---
    print(f"    [GIOU_DBG] Area1 (sample): {area1[:5].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] Area2 (sample): {area2[:5].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] Overlap (sample): {overlap[:5].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] Union (sample): {union[:5].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] IoU (sample): {iou[:5].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] GIoU (sample): {giou[:5].detach().cpu().numpy()}")
    print(f"    [GIOU_DBG] Loss per item (sample): {loss[:5].detach().cpu().numpy()}")
    print("    ---------------- END GIOU_LOSS DEBUG ----------------\n")
    # --- END OF DETAILED DEBUG BLOCK ---

    return loss.sum()


def compute_cls_loss(preds, targets, mask, class_num=80):
    batch_size = targets.shape[0]
    preds_reshape = [p.permute(0, 2, 3, 1).reshape(batch_size, -1, class_num) for p in preds]
    preds = torch.cat(preds_reshape, dim=1)
    
    mask = mask.unsqueeze(-1)
    num_pos = torch.sum(mask.float()).clamp(min=1.0)
    
    if num_pos == 0:
        return torch.tensor(0.0, device=preds.device)
        
    preds_pos = preds[mask.expand_as(preds)].reshape(-1, class_num)
    targets_pos = targets[mask.squeeze(-1)].reshape(-1, 1)
    targets_one_hot = torch.zeros_like(preds_pos).scatter_(1, targets_pos.long() - 1, 1)
    
    loss = focal_loss_from_logits(preds_pos, targets_one_hot)

    # --- ADD THIS DEBUGGING BLOCK ---
    print("\n    --- Inside compute_cls_loss ---")
    print(f"    [CLS_DBG] Num Positives: {num_pos.item()}")
    print(f"    [CLS_DBG] preds_pos shape: {preds_pos.shape}, targets_pos shape: {targets_pos.shape}")
    if num_pos > 0:
        print(f"    [CLS_DBG] Sample Target IDs: {targets_pos[:3].squeeze().tolist()}")
        # We print the one-hot version to verify the mapping is correct
        print(f"    [CLS_DBG] Sample One-Hot Target (sum should be 1.0): {torch.sum(targets_one_hot[:3], dim=-1)}")
        print(f"    [CLS_DBG] Calculated Loss (before norm): {loss.item():.4f}")
    # --- END OF DEBUGGING BLOCK ---

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

    # --- ADD THIS DEBUGGING BLOCK ---
    print("\n    --- Inside compute_cnt_loss ---")
    print(f"    [CNT_DBG] Num Positives: {num_pos.item()}")
    if num_pos > 0:
        print(f"    [CNT_DBG] Sample Centerness Targets: {targets_pos[:5].squeeze().tolist()}")
        print(f"    [CNT_DBG] Sample Centerness Preds (Logits): {preds_pos[:5].squeeze().tolist()}")
        print(f"    [CNT_DBG] Calculated Loss (before norm): {loss.item():.4f}")
    # --- END OF DEBUGGING BLOCK ---

    return loss / num_pos


def compute_reg_loss(preds, targets, mask, coords, mode='giou'):
    preds_reshape = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, 4) for p in preds]
    preds = torch.cat(preds_reshape, dim=1)

    num_pos = torch.sum(mask.float()).clamp(min=1.0)
    if num_pos == 0:
        return torch.tensor(0.0, device=preds.device)

    preds_pos = preds[mask]
    targets_pos = targets[mask]
    coords_pos = coords.unsqueeze(0).repeat(preds.shape[0], 1, 1)[mask]

    if preds_pos.numel() > 0:
        print("\n    ++++++++++++++++++++ BOX RECONSTRUCTION DEBUG ++++++++++++++++++++")
        pred_boxes_x1y1 = coords_pos - preds_pos[:, :2]
        pred_boxes_x2y2 = coords_pos + preds_pos[:, 2:]
        pred_boxes = torch.cat([pred_boxes_x1y1, pred_boxes_x2y2], dim=1)
        target_boxes_x1y1 = coords_pos - targets_pos[:, :2]
        target_boxes_x2y2 = coords_pos + targets_pos[:, 2:]
        target_boxes = torch.cat([target_boxes_x1y1, target_boxes_x2y2], dim=1)
        print(f"    [BOX_RECON] Feature Point Coords (sample):\n{coords_pos[:3].detach().cpu().numpy()}")
        print(f"    [BOX_RECON] Predicted LTRB (sample):\n{preds_pos[:3].detach().cpu().numpy()}")
        print(f"    [BOX_RECON] Reconstructed Predicted Box (x1,y1,x2,y2):\n{pred_boxes[:3].detach().cpu().numpy()}")
        print(f"    [BOX_RECON] Target LTRB (sample):\n{targets_pos[:3].detach().cpu().numpy()}")
        print(f"    [BOX_RECON] Reconstructed Target Box (x1,y1,x2,y2):\n{target_boxes[:3].detach().cpu().numpy()}")
        print("    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    if mode == 'giou':
        loss = giou_loss(pred_boxes, target_boxes)
    else:
        loss = F.l1_loss(preds_pos, targets_pos, reduction='sum')
    
    # --- ADD THIS DEBUGGING PRINT ---
    print(f"\n    --- Inside compute_reg_loss ---")
    print(f"    [REG_DBG] Calculated Loss (before norm): {loss.item():.4f}")
    # --- END OF DEBUGGING PRINT ---

    return loss / num_pos


# -------------------- Loss wrapper --------------------
class LOSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = DefaultConfig if config is None else config
    
    def forward(self, inputs):
        preds, targets ,coords_target= inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        
        # 1. CONVERT JITTOR TO PYTORCH (CPU-safe)
        device = torch.device("cuda" if jt.flags.use_cuda else "cpu")
        cls_logits = [t.to(device) for t in jittor_to_torch(cls_logits)]
        cnt_logits = [t.to(device) for t in jittor_to_torch(cnt_logits)]
        reg_preds = [t.to(device) for t in jittor_to_torch(reg_preds)]
        cls_targets, cnt_targets, reg_targets = jittor_to_torch(cls_targets).to(device), jittor_to_torch(cnt_targets).to(device), jittor_to_torch(reg_targets).to(device)
        coords_target=jittor_to_torch(coords_target).to(device)

        # --- All internal calculations are now in PyTorch ---
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)
        
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos)
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask_pos)
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos,coords_target)
        
        if self.config.add_centerness:
            total_loss = cls_loss + cnt_loss + reg_loss
        else:
            total_loss = cls_loss + reg_loss

        # --- ADD THIS DEBUGGING BLOCK ---
        print(f"  [LOSS-DEBUG] Final PyTorch Losses -> "
              f"cls: {cls_loss.item():.4f}, "
              f"cnt: {cnt_loss.item():.4f}, "
              f"reg: {reg_loss.item():.4f}")
        # --- END OF DEBUGGING BLOCK ---
        
        # 2. CONVERT PYTORCH RESULTS BACK TO JITTOR
        return (torch_to_jittor(cls_loss),
                torch_to_jittor(cnt_loss),
                torch_to_jittor(reg_loss),
                torch_to_jittor(total_loss))