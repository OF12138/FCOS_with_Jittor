import torch
import torch.nn as nn
from .config import DefaultConfig


#--ATSS--
def compute_iou(boxes1, boxes2):
    """
    Computes IoU between two sets of boxes in (x1, y1, x2, y2) format.
    boxes1: (N, 4) tensor of boxes
    boxes2: (M, 4) tensor of boxes
    Returns: (N, M) tensor of IoUs.
    """
    # Calculate intersection area
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Calculate union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    union = area1[:, None] + area2 - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)
    return iou


def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords

class GenTargets(nn.Module):
    def __init__(self,strides,limit_range):
        super().__init__()
        self.strides=strides
        self.limit_range=limit_range
        assert len(strides)==len(limit_range)

    def forward(self,inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits,cnt_logits,reg_preds=inputs[0]
        gt_boxes=inputs[1]
        classes=inputs[2]
        cls_targets_all_level=[]
        cnt_targets_all_level=[]
        reg_targets_all_level=[]
        assert len(self.strides)==len(cls_logits)
        for level in range(len(cls_logits)):
            level_out=[cls_logits[level],cnt_logits[level],reg_preds[level]]
            level_targets=self._gen_level_targets(level_out,gt_boxes,classes,self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
            
        return torch.cat(cls_targets_all_level,dim=1),torch.cat(cnt_targets_all_level,dim=1),torch.cat(reg_targets_all_level,dim=1)

    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits,cnt_logits,reg_preds=out
        batch_size=cls_logits.shape[0]
        class_num=cls_logits.shape[1]
        m=gt_boxes.shape[1]

        cls_logits=cls_logits.permute(0,2,3,1) #[batch_size,h,w,class_num]  
        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2]

        cls_logits=cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]  
        cnt_logits=cnt_logits.permute(0,2,3,1)
        cnt_logits=cnt_logits.reshape((batch_size,-1,1))
        reg_preds=reg_preds.permute(0,2,3,1)
        reg_preds=reg_preds.reshape((batch_size,-1,4))

        h_mul_w=cls_logits.shape[1]

        x=coords[:,0]
        y=coords[:,1]
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[batch_size,h*w,m,4]

        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[batch_size,h*w,m]

        off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
        off_max=torch.max(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]

        mask_in_gtboxes=off_min>0
        mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])

        radiu=stride*sample_radiu_ratio
        gt_center_x=(gt_boxes[...,0]+gt_boxes[...,2])/2
        gt_center_y=(gt_boxes[...,1]+gt_boxes[...,3])/2
        c_l_off=x[None,:,None]-gt_center_x[:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off=y[None,:,None]-gt_center_y[:,None,:]
        c_r_off=gt_center_x[:,None,:]-x[None,:,None]
        c_b_off=gt_center_y[:,None,:]-y[None,:,None]
        c_ltrb_off=torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[batch_size,h*w,m,4]
        c_off_max=torch.max(c_ltrb_off,dim=-1)[0]
        mask_center=c_off_max<radiu

        mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]

        areas[~mask_pos]=99999999
        areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]
        reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))#[batch_size,h*w,4]

        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]#[batch_size,h*w,m]
        cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]

        assert reg_targets.shape==(batch_size,h_mul_w,4)
        assert cls_targets.shape==(batch_size,h_mul_w,1)
        assert cnt_targets.shape==(batch_size,h_mul_w,1)

        #process neg coords
        mask_pos_2=mask_pos.long().sum(dim=-1)#[batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2=mask_pos_2>=1
        assert mask_pos_2.shape==(batch_size,h_mul_w)
        cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]
        cnt_targets[~mask_pos_2]=-1
        reg_targets[~mask_pos_2]=-1
        
        return cls_targets,cnt_targets,reg_targets

    def _gen_level_targets_atss(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        cls_logits,cnt_logits,reg_preds=out
        batch_size=cls_logits.shape[0]
        class_num=cls_logits.shape[1]
        m=gt_boxes.shape[1]

        cls_logits=cls_logits.permute(0,2,3,1)
        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)
        
        cls_logits=cls_logits.reshape((batch_size,-1,class_num))
        cnt_logits=cnt_logits.permute(0,2,3,1).reshape((batch_size,-1,1))
        reg_preds=reg_preds.permute(0,2,3,1).reshape((batch_size,-1,4))

        h_mul_w=cls_logits.shape[1]

        x=coords[:,0]
        y=coords[:,1]
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)

        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])
        off_min=torch.min(ltrb_off,dim=-1)[0]
        mask_in_gtboxes=off_min>0
        
        # ---- START OF CHANGES: Simplified ATSS with Bug Fix ----
        ious_all = []
        for b in range(batch_size):
            # Filter out padded GT boxes for this image
            gt_boxes_b = gt_boxes[b]
            valid_gt_mask = gt_boxes_b[:, 0] != -1
            gt_boxes_b = gt_boxes_b[valid_gt_mask]
            ltrb_off_b = ltrb_off[b][:, valid_gt_mask] # Shape: [h*w, num_valid_gt, 4]
            
            if gt_boxes_b.shape[0] == 0: # If no GT boxes in this image
                ious_all.append(torch.zeros(h_mul_w, m, device=coords.device))
                continue
                
            # BUG FIX: Corrected broadcasting for creating predicted boxes
            # Coords shape: [h*w, 2] -> unsqueeze to [h*w, 1, 2]
            # ltrb_off_b shape: [h*w, num_valid_gt, 4]
            coords_unsqueezed = coords.unsqueeze(1)
            pred_boxes_x1y1 = coords_unsqueezed - ltrb_off_b[..., :2] # Result: [h*w, num_valid_gt, 2]
            pred_boxes_x2y2 = coords_unsqueezed + ltrb_off_b[..., 2:] # Result: [h*w, num_valid_gt, 2]
            pred_boxes_for_gt = torch.cat([pred_boxes_x1y1, pred_boxes_x2y2], dim=-1) # Result: [h*w, num_valid_gt, 4]

            # Vectorized IoU calculation
            # gt_boxes_b shape: [num_valid_gt, 4]
            # pred_boxes_for_gt shape: [h*w, num_valid_gt, 4]
            # We want to compare each GT with its corresponding column of predicted boxes
            ious_b = compute_iou(
                pred_boxes_for_gt.reshape(-1, 4), # Shape: [h*w * num_valid_gt, 4]
                gt_boxes_b
            ).reshape(h_mul_w, gt_boxes_b.shape[0], gt_boxes_b.shape[0]) # Shape: [h*w, num_valid_gt, num_valid_gt]

            # We only need the diagonal: IoU of GT_i with predictions for GT_i
            ious_b = torch.diagonal(ious_b, dim1=1, dim2=2) # Shape: [h*w, num_valid_gt]
            
            # Pad back to original size 'm'
            padded_ious = torch.zeros(h_mul_w, m, device=coords.device)
            padded_ious[:, valid_gt_mask] = ious_b
            ious_all.append(padded_ious)

        ious = torch.stack(ious_all, dim=0)
        
        ious_valid = ious * mask_in_gtboxes
        iou_mean = torch.sum(ious_valid, dim=1, keepdim=True) / torch.sum(mask_in_gtboxes, dim=1, keepdim=True).clamp(min=1)
        iou_std = torch.sqrt(torch.sum(torch.pow(ious_valid - iou_mean, 2) * mask_in_gtboxes, dim=1, keepdim=True) / torch.sum(mask_in_gtboxes, dim=1, keepdim=True).clamp(min=1))
        
        atss_threshold = iou_mean + iou_std
        mask_atss = ious >= atss_threshold
        mask_pos = mask_in_gtboxes & mask_atss
        # ---- END OF CHANGES ----

        areas[~mask_pos]=99999999
        areas_min_ind=torch.min(areas,dim=-1)[1]
        reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))

        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]
        cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)

        mask_pos_2=mask_pos.long().sum(dim=-1)>=1
        cls_targets[~mask_pos_2]=0
        cnt_targets[~mask_pos_2]=-1
        reg_targets[~mask_pos_2]=-1
        
        return cls_targets,cnt_targets,reg_targets
  
        
def compute_cls_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    preds_reshape=[]
    class_num=preds[0].shape[1]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,class_num])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)#[batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2]==targets.shape[:2]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index]#[sum(_h*_w),class_num]
        target_pos=targets[batch_index]#[sum(_h*_w),1]
        target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
        loss.append(focal_loss_from_logits(pred_pos,target_pos).view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_cnt_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),1]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,]
        assert len(pred_pos.shape)==1
        loss.append(nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_reg_loss(preds,targets,mask,mode='giou'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos=torch.sum(mask,dim=1).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        assert len(pred_pos.shape)==2
        if mode=='iou':
            loss.append(iou_loss(pred_pos,target_pos).view(1))
        elif mode=='giou':
            loss.append(giou_loss(pred_pos,target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def iou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt=torch.min(preds[:,:2],targets[:,:2])
    rb=torch.min(preds[:,2:],targets[:,2:])
    wh=(rb+lt).clamp(min=0)
    overlap=wh[:,0]*wh[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    iou=overlap/(area1+area2-overlap)
    loss=-iou.clamp(min=1e-6).log()
    return loss.sum()

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds=preds.sigmoid()
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*targets+(1.0-alpha)*(1.0-targets)
    loss=-w*torch.pow((1.0-pt),gamma)*pt.log()
    return loss.sum()




class LOSS(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds,targets=inputs
        cls_logits,cnt_logits,reg_preds=preds
        cls_targets,cnt_targets,reg_targets=targets
        mask_pos=(cnt_targets>-1).squeeze(dim=-1)# [batch_size,sum(_h*_w)]
        cls_loss=compute_cls_loss(cls_logits,cls_targets,mask_pos).mean()#[]
        cnt_loss=compute_cnt_loss(cnt_logits,cnt_targets,mask_pos).mean()
        reg_loss=compute_reg_loss(reg_preds,reg_targets,mask_pos).mean()
        if self.config.add_centerness:
            total_loss=cls_loss+cnt_loss+reg_loss
            return cls_loss,cnt_loss,reg_loss,total_loss
        else:
            total_loss=cls_loss+reg_loss+cnt_loss*0.0
            return cls_loss,cnt_loss,reg_loss,total_loss


#--test--




if __name__ == "__main__":
    # 1. Configuration and Layer Instantiation
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
    gen_targets_layer = GenTargets(strides=strides, limit_range=limit_range)

    # 2. Create Mock/Fake Data to Simulate Real Inputs
    batch_size = 2
    class_num = 80
    num_gt = 5  # Max number of ground truth boxes per image in a batch
    
    # Mock model predictions from the FPN
    fpn_out_shapes = [(100, 168), (50, 84), (25, 42), (13, 21), (7, 11)]  # Example H, W for each FPN level
    cls_logits = [torch.randn(batch_size, class_num, h, w) for h, w in fpn_out_shapes]
    cnt_logits = [torch.randn(batch_size, 1, h, w) for h, w in fpn_out_shapes]
    reg_preds = [torch.abs(torch.randn(batch_size, 4, h, w)) for h, w in fpn_out_shapes]
    preds = [cls_logits, cnt_logits, reg_preds]

    # Mock ground truth boxes and classes
    gt_boxes = torch.rand(batch_size, num_gt, 4) * 200  # Random boxes with coords up to 200
    gt_boxes[..., 2:] += gt_boxes[..., :2] + 20 # Ensure x2 > x1 and y2 > y1, with a minimum size of 20
    
    # Simulate padding by setting the last box of the first image to -1
    gt_boxes[0, -1, :] = -1
    
    classes = torch.randint(1, class_num + 1, (batch_size, num_gt))
    classes[0, -1] = -1

    # 3. Run the Test and Print Results
    print("--- Running GenTargets Test with Mock Data ---")
    try:
        # Call the forward method to generate targets
        cls_targets, cnt_targets, reg_targets = gen_targets_layer([preds, gt_boxes, classes])
        
        # 4. Print Output Shapes and Info
        print("\n✅ Successfully generated targets!")
        print(f"   cls_targets shape: {cls_targets.shape}")
        print(f"   cnt_targets shape: {cnt_targets.shape}")
        print(f"   reg_targets shape: {reg_targets.shape}")
        
        # Check if any positive samples were found (important sanity check)
        num_pos = (cnt_targets > -1).sum().item()
        print(f"\nTotal positive samples found across batch: {num_pos}")

        if num_pos == 0:
            print("⚠️ Warning: No positive samples were assigned. This can happen with random data,")
            print("   but if it persists with real data, your ATSS thresholds might be too strict.")

    except Exception as e:
        print(f"\n❌ ERROR ---")
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        print("\n--- Debugging Tips ---")
        print("1. Double-check tensor shapes inside the `_gen_level_targets` function.")
        print("2. The error might be due to incorrect broadcasting or an edge case (e.g., no GT boxes).")
    

    


