import jittor as jt
import jittor.nn as nn
from .config import DefaultConfig

def coords_fmap2orig(feature, stride):
    """
    Convert feature map coordinates to image coordinates (pixel units).
    feature: [B, C, H, W] or [C, H, W]
    stride: int, stride of this feature map relative to the image
    return: coords [HW, 2] in (x, y) order, aligned with image pixel centers
    """
    h, w = feature.shape[-2:]   # always take last two dims
    shifts_x = jt.arange(0, w, dtype=jt.float32) + 0.5
    shifts_y = jt.arange(0, h, dtype=jt.float32) + 0.5
    shifts_x = shifts_x * stride
    shifts_y = shifts_y * stride
    shift_y, shift_x = jt.meshgrid(shifts_y, shifts_x)  # [H,W]
    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])
    coords = jt.stack([shift_x, shift_y], dim=-1)  # [HW,2]
    return coords


# -------------------- Target generation --------------------
class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        #assert len(self.strides) == len(self.limit_range) ###

    def execute(self, inputs):
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes, classes = inputs[1], inputs[2]

        cls_targets_all, cnt_targets_all, reg_targets_all = [], [], []
        for lvl, stride in enumerate(self.strides):
            cls_t, cnt_t, reg_t = self._gen_level_targets(
                cls_logits[lvl], gt_boxes, classes,
                stride, self.limit_range[lvl]
            )
            cls_targets_all.append(cls_t)
            cnt_targets_all.append(cnt_t)
            reg_targets_all.append(reg_t)

        return (jt.concat(cls_targets_all, dim=1),
                jt.concat(cnt_targets_all, dim=1),
                jt.concat(reg_targets_all, dim=1))

    def _gen_level_targets(self, feature_map, gt_boxes, classes, stride, limit_range, sample_radius_ratio=3.0 ):
        B, _, H, W = feature_map.shape
        HW = H * W
        min_range, max_range = float(limit_range[0]), float(limit_range[1])

        # coords on original image
        shifts_x = jt.arange(0, W * stride, stride, dtype="float32") + stride/2.0
        shifts_y = jt.arange(0, H * stride, stride, dtype="float32") + stride/2.0
        shift_y, shift_x = jt.meshgrid(shifts_y, shifts_x)
        coords = jt.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1)  # (HW, 2)

        cls_list, cnt_list, reg_list = [], [], []
        INF = 1e10

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

            # expand
            coords_exp = coords.unsqueeze(1).repeat(1, m, 1)        # (HW, m, 2)
            boxes_exp = boxes_valid.unsqueeze(0).repeat(HW, 1, 1)   # (HW, m, 4)

            x = coords_exp[..., 0]
            y = coords_exp[..., 1]
            l = x - boxes_exp[..., 0]
            t = y - boxes_exp[..., 1]
            r = boxes_exp[..., 2] - x
            b_ = boxes_exp[..., 3] - y
            ltrb = jt.stack([l, t, r, b_], dim=-1)  # (HW, m, 4)

            # inside-box mask
            off_min = ltrb.min(dim=-1)[0]  # (HW, m)
            mask_in_box = off_min >= -1

                # 1. Relax mask_in_box criteria
            # Original: mask_in_box = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            # New: Allow some tolerance outside the box
            #tolerance = stride * 0.5  # Add small tolerance
            #mask_in_box = (x >= (xmin - tolerance)) & (x <= (xmax + tolerance)) & \
                        #(y >= (ymin - tolerance)) & (y <= (ymax + tolerance))

            # range mask (max offset)
            off_max = ltrb.max(dim=-1)[0]  # (HW, m)
            mask_in_level = (off_max > min_range) & (off_max <= max_range)

            # center sample mask
            cx = (boxes_valid[:, 0] + boxes_valid[:, 2]) / 2.0  # (m,)
            cy = (boxes_valid[:, 1] + boxes_valid[:, 3]) / 2.0  # (m,)
            cx_exp = cx.unsqueeze(0).repeat(HW, 1)
            cy_exp = cy.unsqueeze(0).repeat(HW, 1)
            c_l = x - cx_exp
            c_t = y - cy_exp
            c_r = cx_exp - x
            c_b = cy_exp - y
            c_off_max = jt.stack([c_l, c_t, c_r, c_b], dim=-1).max(dim=-1)[0]  # (HW,m)
            radius = stride * sample_radius_ratio
            mask_center = c_off_max < radius

            pos_mask = mask_in_box & mask_in_level #& mask_center  # (HW, m)

            if pos_mask.sum() == 0:
                radius_relaxed = radius * 2.0
                mask_center_relaxed = c_off_max < radius_relaxed
                pos_mask = mask_in_box & mask_in_level & mask_center_relaxed
                print(f"Warning: No positives with strict center sampling (stride={stride}), using relaxed radius {radius_relaxed}")



            # area per gt broadcasted
            areas = ((boxes_valid[:, 2] - boxes_valid[:, 0]) * (boxes_valid[:, 3] - boxes_valid[:, 1]))  # (m,)
            areas_exp = areas.unsqueeze(0).repeat(HW, 1)  # (HW, m)
            areas_valid = jt.where(pos_mask, areas_exp, jt.ones_like(areas_exp) * INF)  # (HW, m)

            # --- DEBUG: add right after pos_mask is computed ---
            if b == 0:
                try:
                    # Basic shapes / counts
                    print(f" stride={stride} HxW={H}x{W} boxes_valid.shape={boxes_valid.shape} m={m}")
                    print(" sample boxes_valid (first 5):",
                        boxes_valid[:5].numpy().tolist() if boxes_valid.numel() > 0 else [])
                    # mask sums
                    mib = int(mask_in_box.sum().item())
                    mil = int(mask_in_level.sum().item())
                    mc  = int(mask_center.sum().item())
                    pm  = int(pos_mask.sum().item())
                    print(" mask_in_box.sum=", mib, " mask_in_level.sum=", mil, " mask_center.sum=", mc, " pos_mask.sum=", pm)

                    # area stats
                    if areas_exp.numel() > 0 and m > 0:
                        areas_np = areas_exp.reshape(-1).numpy()
                        print(" area min/max:", float(areas_np.min()), float(areas_np.max()))
                    # show a few coords vs box centers for manual check
                    coords_np = coords.numpy() if coords.numel() > 0 else None
                    if coords_np is not None:
                        print(" coords sample (first 5):", coords_np[:5].tolist())
                        # show first box center
                        if boxes_valid.numel() > 0:
                            first_box = boxes_valid[0].numpy().tolist()
                            cx = (first_box[0] + first_box[2]) / 2.0
                            cy = (first_box[1] + first_box[3]) / 2.0
                            print(" first_box:", first_box, " center:", [cx, cy])
                except Exception as e:
                    print("DEBUG PRINT ERROR:", e)


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
                lr_max = jt.maximum(reg_selected[:, 0], reg_selected[:, 2])
                tb_min = jt.minimum(reg_selected[:, 1], reg_selected[:, 3])
                tb_max = jt.maximum(reg_selected[:, 1], reg_selected[:, 3])
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
