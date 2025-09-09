import jittor as jt
import jittor.nn as nn
from .config import DefaultConfig

#helper function 
def coords_fmap2orig(feature, stride):
    """
    Convert feature map coordinates to image coordinates (pixel units).
    feature: [B, C, H, W] or [C, H, W]
    stride: int, stride of this feature map relative to the image
    return: coords [HW, 2] in (x, y) order, aligned with image pixel centers
    """
    h, w = feature.shape[-2:]   # always take last two dims
    shifts_x = jt.arange(0, w, dtype=jt.float32) + 0.5 #jt.arange(start, end) 
    shifts_y = jt.arange(0, h, dtype=jt.float32) + 0.5 # the + 0.5 convert the top-left to the center 
    shifts_x = shifts_x * stride #the stride means the ratio(of pixels) from input to feature map
    shifts_y = shifts_y * stride
    shift_x,shift_y = jt.meshgrid(shifts_x,shifts_y)  # [H,W] #convert the 1D tensor to 2D tensor by copy 
    shift_x = shift_x.reshape([-1]) #flatten the 2D tensor
    shift_y = shift_y.reshape([-1])
    coords = jt.stack([shift_x, shift_y], dim=-1)  # [HW,2] #make them into pairs
    return coords


# -------------------- Target generation --------------------
class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range #store the size range for each FPN level
        assert len(self.strides) == len(self.limit_range) 

    #the target generation
    def execute(self, inputs):
        #unpack the model's predictions
        cls_logits, cnt_logits, reg_preds = inputs[0] 
        gt_boxes, classes, batch_scales = inputs[1], inputs[2] ,inputs[3]
        cls_targets_all, cnt_targets_all, reg_targets_all = [], [], []

        for lvl, stride in enumerate(self.strides):
            cls_t, cnt_t, reg_t = self._gen_level_targets(
                cls_logits[lvl], gt_boxes, classes, batch_scales,
                stride, self.limit_range[lvl]
            )
            cls_targets_all.append(cls_t)
            cnt_targets_all.append(cnt_t)
            reg_targets_all.append(reg_t)

        return (jt.concat(cls_targets_all, dim=1),
                jt.concat(cnt_targets_all, dim=1),
                jt.concat(reg_targets_all, dim=1))  

    #applies the matching rules for each single FPN level
    def _gen_level_targets(self, feature_map, gt_boxes, classes, batch_scales, stride, limit_range, sample_radius_ratio=1.5 ):
        B, _, H, W = feature_map.shape
        HW = H * W
        #the valid object size range for this level
        min_range, max_range = float(limit_range[0]), float(limit_range[1])

        #--- coordinate generation---
        #This block generates the [x,y] coordinates for every point in the feature map
        shifts_x = jt.arange(0, W * stride, stride, dtype="float32") + stride / 2.0  # [W,]
        shifts_y = jt.arange(0, H * stride, stride, dtype="float32") + stride / 2.0  # [H,]
        shift_x, shift_y = jt.meshgrid(shifts_x, shifts_y)
        coords_original = jt.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1)  # [HW, 2]

        cls_list, cnt_list, reg_list = [], [], []
        INF = 1e10

        #loop through each image in the batch
        for b in range(B):
            boxes_b, cls_b = gt_boxes[b], classes[b]
            valid_mask = (cls_b >= 0)
            if valid_mask.sum().item() == 0:
                cls_list.append(jt.zeros((HW,1), dtype="int32"))
                cnt_list.append(jt.ones((HW,1), dtype="float32") * -1.0)
                reg_list.append(jt.ones((HW,4), dtype="float32") * -1.0)
                continue

            boxes_valid = boxes_b[valid_mask]                # (m,4)
            cls_valid = cls_b[valid_mask].int32()            # (m,)
            m = int(boxes_valid.shape[0])
            if m == 0:
                cls_list.append(jt.zeros((HW,1), dtype="int32"))
                cnt_list.append(jt.ones((HW,1), dtype="float32") * -1.0)
                reg_list.append(jt.ones((HW,4), dtype="float32") * -1.0)
                continue

            # --- Debug Block ---
            print(f"\n--- COORDINATE SYSTEM CHECK (Stride: {stride}) ---")
            print(f"  Feature Grid X Range: [{coords_original[:, 0].min().item():.2f}, {coords_original[:, 0].max().item():.2f}]")
            print(f"  Feature Grid Y Range: [{coords_original[:, 1].min().item():.2f}, {coords_original[:, 1].max().item():.2f}]")
            # Print the first GT box for comparison
            print(f"  First GT Box (x1, y1, x2, y2): [{boxes_valid[0][0]:.2f}, {boxes_valid[0][1]:.2f}, {boxes_valid[0][2]:.2f}, {boxes_valid[0][3]:.2f}]")
            print(f"--------------------------------------------------\n")
            

            coords_b = coords_original  # [HW,2] (scaled space for current image)
            coords_exp = coords_b.unsqueeze(1).repeat(1, m, 1)  # [HW,m,2]
            boxes_exp = boxes_valid.unsqueeze(0).repeat(HW, 1, 1)  # [HW,m,4]


            x = coords_exp[..., 0]
            y = coords_exp[..., 1]
            l = x - boxes_exp[..., 0]
            t = y - boxes_exp[..., 1]
            r = boxes_exp[..., 2] - x
            b_ = boxes_exp[..., 3] - y
            ltrb = jt.stack([l, t, r, b_], dim=-1)  # (HW, m, 4)


            # ... after ltrb calculation ...
            if b == 0: # only print for first batch item to avoid spam
                # print("ltrb for first GT box:", ltrb[0, 0, :].numpy()) # uncomment if needed
                # print("off_max for first GT box:", ltrb.max(dim=-1)[0][0, 0].item()) # only for first feature map point and first GT
                print(f"  Stride {stride}, limit_range: {limit_range}")
                print(f"GT Box (resized space): {boxes_valid[0].tolist()}")  # Should match scaled boxes in logs
                print(f"Feature Points (resized space): {coords_b[:5].tolist()}")  # Should overlap with GT boxes
                print(f"Offsets (l,t,r,b): [{l[0,0].item()}, {t[0,0].item()}, {r[0,0].item()}, {b_[0,0].item()}]")
                # print(f"  off_max for all points, first GT: {ltrb.max(dim=-1)[0][:,0].numpy()}") # print for first GT and all feature points
                # print(f"  off_max (min, max) overall for batch {b}, stride {stride}: ({off_max.min().item():.2f}, {off_max.max().item():.2f})")

            # Fix 3: Correct mask_in_level (use valid limit_range [0, max])
            # --------------------------
            off_min = ltrb.min(dim=-1)[0]  # [HW,m]
            off_max = ltrb.max(dim=-1)[0]  # [HW,m]
            mask_in_box = off_min >= 0  # Strictly inside (standard FCOS, no -1 tolerance)
            mask_in_level = (off_max >= min_range) & (off_max <= max_range)  # Use >=0, not >-1

            # center sample mask (apply only to positive samples later)
            # --------------------------
            # Fix 4: Correct mask_center (scaled center coordinates)
            # --------------------------
            cx = (boxes_valid[:, 0] + boxes_valid[:, 2]) / 2.0  # [m,] (scaled GT center)
            cy = (boxes_valid[:, 1] + boxes_valid[:, 3]) / 2.0  # [m,]
            cx_exp = cx.unsqueeze(0).repeat(HW, 1)  # [HW,m]
            cy_exp = cy.unsqueeze(0).repeat(HW, 1)  # [HW,m]
            c_l = x - cx_exp
            c_t = y - cy_exp
            c_r = cx_exp - x
            c_b = cy_exp - y
            c_off_max = jt.stack([c_l, c_t, c_r, c_b], dim=-1).max(dim=-1)[0]  # [HW,m]
            radius = stride * sample_radius_ratio  # 8*1.5=12 (standard)
            mask_center = c_off_max < radius


            # Combine masks
            # Note: The original FCOS paper's central sampling is usually applied *after*
            # determining candidates from mask_in_box and mask_in_level.
            # Your current `pos_mask` definition (`mask_in_box & mask_in_level`) is correct for the first stage.
            pos_mask = mask_in_box & mask_in_level & mask_center # (HW, m)


            # --- Relaxed center sampling logic (already good) ---
            if pos_mask.sum() == 0:
                # If no positives, relax the center sampling constraint
                # Note: This relaxation is *only* applied if the initial combined mask is empty.
                # If you want to use center sampling as a hard constraint, it should be part of pos_mask.
                # Given your debug output, this relaxation is essential.
                radius_relaxed = radius * 2.0
                mask_center_relaxed = c_off_max < radius_relaxed
                pos_mask = mask_in_box & mask_in_level & mask_center_relaxed
                print(f"Warning: No positives with strict center sampling (stride={stride}), using relaxed radius {radius_relaxed}")
            else:
                # Only apply center sampling if there are already some positive candidates
                # This ensures that center sampling refines existing positives, rather than creating new ones
                # from outside the box/level limits.
                pos_mask = pos_mask & mask_center
            # ---------------------------------------------------
            ltrb = ltrb.clamp(min_v=1e-5)  # Prevent 0/negative offsets (causes NaN in loss)

            # ... rest of the function (area calculation, argmin, scatter) ...
            # The rest of your target generation logic for areas, argmin, and scatter operations looks generally correct
            # after the pos_mask is properly identified.

            # area per gt broadcasted
            areas = ((boxes_valid[:, 2] - boxes_valid[:, 0]) * (boxes_valid[:, 3] - boxes_valid[:, 1]))  # (m,)
            areas_exp = areas.unsqueeze(0).repeat(HW, 1)  # (HW, m)
            areas_valid = jt.where(pos_mask, areas_exp, jt.ones_like(areas_exp) * INF)  # (HW, m)

            # ... (DEBUG print block is fine) ...

            # get index of min area per location using argmin
            argmin_res = jt.argmin(areas_valid, dim=1)
            if isinstance(argmin_res, tuple):
                min_inds = argmin_res[1].int32()
            else:
                min_inds = argmin_res.int32()

            # ---- DEFENSIVE CLAMP: ensure indices in [0, m-1] before any gather/getitem ----
            # convert to int32 and clamp
            min_inds = min_inds.astype(jt.int32)
            min_inds = jt.minimum(jt.maximum(min_inds, jt.zeros_like(min_inds)), jt.ones_like(min_inds) * (m - 1))
            # ---------------------------------------------------------------------------

            # now get min values by gather (safe because indices clamped)
            min_vals = areas_valid.gather(1, min_inds.unsqueeze(1)).squeeze(1)  # (HW,)

            # prepare default outputs
            cls_target = jt.zeros((HW, 1), dtype="int32")
            cnt_target = jt.ones((HW, 1), dtype="float32") * -1.0
            reg_target = jt.ones((HW, 4), dtype="float32") * -1.0

            # positive location mask (found a real gt)
            pos_loc_mask = min_vals < INF  # (HW,)

            # ... after calculating all masks ...
            print(f"Batch {b}, Stride {stride}:")
            print(f"  mask_in_box sum: {mask_in_box.sum().item()}")
            print(f"  mask_in_level sum: {mask_in_level.sum().item()}")
            print(f"  mask_center sum (ratio {sample_radius_ratio}): {mask_center.sum().item()}")
            print(f"  Final pos_loc_mask sum (after argmin/INF filter): {pos_loc_mask.sum().item()}")
            print("-" * 20)


            if pos_loc_mask.sum().item() > 0:
                # Only operate on positive locations to avoid any accidental use of clamped indices
                pos_indices = pos_loc_mask.nonzero().squeeze()
                if pos_indices.numel() == 0:
                    # defensive fallback
                    cls_list.append(cls_target); cnt_list.append(cnt_target); reg_list.append(reg_target)
                    continue

                # gather reg values for selected gt per positive location
                # build per-positive indices safely
                min_inds_pos = min_inds[pos_indices]  # (P,)
                # build gather indices aligned with ltrb shape: (P,1,4)
                idx_for_gather = min_inds_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4)
                # gather from ltrb at positive row positions
                reg_selected_all = ltrb.gather(1, idx_for_gather)  # (HW,1,4) but we'll index by pos_indices
                reg_selected = reg_selected_all[pos_indices].squeeze(1)  # (P,4)

                # class selected
                cls_selected = cls_valid.gather(0, min_inds_pos).reshape(-1, 1).int32()  # (P,1)

                # centerness calculation for positives
                lr_min = jt.minimum(reg_selected[:, 0], reg_selected[:, 2])
                lr_max = jt.maximum(reg_selected[:, 0], reg_selected[:, 2]).clamp(min_v=1e-5)
                tb_min = jt.minimum(reg_selected[:, 1], reg_selected[:, 3])
                tb_max = jt.maximum(reg_selected[:, 1], reg_selected[:, 3]).clamp(min_v=1e-5)
                centerness_all = jt.sqrt((lr_min * tb_min) / (lr_max * tb_max + 1e-10)).unsqueeze(1)  # (P,1)

                # write selected values into full tensors at positive positions
                cls_target = cls_target  # (HW,1)
                reg_target = reg_target  # (HW,4)
                cnt_target = cnt_target  # (HW,1)

                # assign values using boolean indexing: safer than gather with questionable indices
                cls_target = cls_target.scatter(0, pos_indices.unsqueeze(1).astype(jt.int32), cls_selected)
                reg_target = reg_target.scatter(0, pos_indices.unsqueeze(1).repeat(1,4).astype(jt.int32), reg_selected)
                cnt_target = cnt_target.scatter(0, pos_indices.unsqueeze(1).astype(jt.int32), centerness_all)

            cls_list.append(cls_target)
            cnt_list.append(cnt_target)
            reg_list.append(reg_target)

        return jt.stack(cls_list, 0), jt.stack(cnt_list, 0), jt.stack(reg_list, 0)

# -------------------- Loss functions --------------------
def compute_cls_loss(preds, targets, mask):
    """
    preds: list of 5 levels [B, C, H, W]
    targets: [B, sum(HW), 1]
    mask: [B, sum(HW)]
    """
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)

    # number of positives per batch, clamp to avoid div/0
    num_pos = jt.sum(mask.float(), dims=[1, 2]).clamp(min_v=1)

    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)            # [B, H, W, C]
        pred = pred.reshape(batch_size, -1, class_num)
        preds_reshape.append(pred)
    preds = jt.concat(preds_reshape, dim=1)        # [B, sum(HW), C]
    assert preds.shape[:2] == targets.shape[:2]

    loss = []
    for b in range(batch_size):
        target_pos = targets[b]                    # [sum(HW), 1]
        if mask[b].sum().item() == 0:              # <-- guard
            loss.append(jt.zeros(1))
            continue

        # one-hot encode
        target_pos = (jt.arange(0, class_num)[None, :] == target_pos).float()
        pred_pos = preds[b]                        # [sum(HW), C]
        loss.append(focal_loss_from_logits(pred_pos, target_pos).reshape(1))

    return jt.concat(loss, dim=0) / num_pos        # [B,]


def compute_cnt_loss(preds, targets, mask):
    """
    preds: list of 5 levels [B, 1, H, W]
    targets: [B, sum(HW), 1]
    mask: [B, sum(HW)]
    """
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)

    num_pos = jt.sum(mask.float(), dims=[1, 2]).clamp(min_v=1)

    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)            # [B, H, W, 1]
        pred = pred.reshape(batch_size, -1, c)
        preds_reshape.append(pred)
    preds = jt.concat(preds_reshape, dim=1)        # [B, sum(HW), 1]
    assert preds.shape == targets.shape

    loss = []
    for b in range(batch_size):
        pred_pos = preds[b][mask[b]]
        target_pos = targets[b][mask[b]]

        if pred_pos.numel() == 0:                  # <-- guard
            loss.append(jt.zeros(1))
        else:
            loss.append(
                nn.binary_cross_entropy_with_logits(
                    output=pred_pos, target=target_pos, size_average=False
                ).reshape(1)
            )
    return jt.concat(loss, dim=0) / num_pos


def compute_reg_loss(preds, targets, mask, mode="giou"):
    """
    preds: list of 5 levels [B, 4, H, W]
    targets: [B, sum(HW), 4]
    mask: [B, sum(HW)]
    """
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    num_pos = jt.sum(mask.float(), dim=1).clamp(min_v=1)

    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)            # [B, H, W, 4]
        pred = pred.reshape(batch_size, -1, c)
        preds_reshape.append(pred)
    preds = jt.concat(preds_reshape, dim=1)        # [B, sum(HW), 4]
    assert preds.shape == targets.shape

    loss = []
    for b in range(batch_size):
        pred_pos = preds[b][mask[b]]
        target_pos = targets[b][mask[b]]

        if pred_pos.numel() == 0:                  # <-- guard
            loss.append(jt.zeros(1))
        else:
            # Clamp to avoid extreme values (e.g., negative offsets)
            pred_pos = pred_pos.clamp(min_v=1e-5, max_v=1e5)
            target_pos = target_pos.clamp(min_v=1e-5, max_v=1e5)
            if mode == "iou":
                loss.append(iou_loss(pred_pos, target_pos).reshape(1))
            elif mode == "giou":
                loss.append(giou_loss(pred_pos, target_pos).reshape(1))
            else:
                raise NotImplementedError("reg loss only supports ['iou','giou']")
    return jt.concat(loss, dim=0) / num_pos


def iou_loss(preds, targets):
    lt = jt.minimum(preds[:,:2], targets[:,:2])
    rb = jt.minimum(preds[:,2:], targets[:,2:])
    wh = (rb+lt).clamp(min_v=0)
    overlap = wh[:,0]*wh[:,1]
    area1 = (preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2 = (targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    iou = overlap/(area1+area2-overlap+1e-10)
    return -(iou.clamp(min_v=1e-6)).log().sum()

def giou_loss(preds, targets):
    lt_min = jt.minimum(preds[:,:2], targets[:,:2])
    rb_min = jt.minimum(preds[:,2:], targets[:,2:])
    wh_min = (rb_min+lt_min).clamp(min_v=0)
    overlap = wh_min[:,0]*wh_min[:,1]
    area1 = (preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2 = (targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union = area1+area2-overlap
    iou = overlap/(union+1e-10)

    lt_max = jt.maximum(preds[:,:2], targets[:,:2])
    rb_max = jt.maximum(preds[:,2:], targets[:,2:])
    wh_max = (rb_max+lt_max).clamp(min_v=0)
    G_area = wh_max[:,0]*wh_max[:,1]
    giou = iou-(G_area-union)/(G_area+1e-10)
    return (1-giou).sum()

def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    preds = preds.sigmoid()
    pt = preds*targets + (1-preds)*(1-targets)
    pt = pt.clamp(min_v=1e-12, max_v=1 - 1e-12)  # Prevent log(0) or log(1)
    w = alpha*targets + (1-alpha)*(1-targets)
    return (-w * (1-pt).pow(gamma) * (pt+1e-12).log()).sum()

# -------------------- Loss wrapper --------------------
class LOSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = DefaultConfig if config is None else config

    def execute(self, inputs):
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        mask = (cnt_targets > -1)
        if mask.ndim==3: mask = mask.squeeze(-1)

        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask).mean()
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask).mean()
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask).mean()

        if self.config.add_centerness:
            total = cls_loss+cnt_loss+reg_loss
        else:
            total = cls_loss+reg_loss

        # ---- DEBUG ----
        try:
            print(
                "mask.sum():", mask.sum().item(),
                "cls_targets unique:", jt.unique(cls_targets).numpy()[:10],
                "cnt_targets min/max:", float(cnt_targets.min().item()), float(cnt_targets.max().item()),
                "reg_targets mean:", float(reg_targets.mean().item()),
            )
        except Exception as e:
            print("[DEBUG PRINT FAILED]", e)
        # --------------

        return cls_loss, cnt_loss, reg_loss, total
