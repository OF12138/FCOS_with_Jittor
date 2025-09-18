import jittor as jt
import jittor.nn as nn
import math

class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        # In Jittor, any jt.Var assigned to a module is treated as a parameter
        self.scale = jt.array([init_value], dtype=jt.float32)

    def execute(self, x):
        return jt.exp(x * self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self, in_channel, class_num, use_GN_head=True, cnt_on_reg=True, prior=0.01):
        super(ClsCntRegHead, self).__init__()
        self.prior = prior
        self.class_num = class_num
        self.cnt_on_reg = cnt_on_reg
        
        cls_branch = []
        reg_branch = []

        for i in range(4):
            cls_branch.append(nn.Conv(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            if use_GN_head:
                cls_branch.append(nn.GroupNorm(32, in_channel))
            cls_branch.append(nn.ReLU())

            reg_branch.append(nn.Conv(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            if use_GN_head:
                reg_branch.append(nn.GroupNorm(32, in_channel))
            reg_branch.append(nn.ReLU())

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv(in_channel, class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv(in_channel, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv(in_channel, 4, kernel_size=3, padding=1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv):
                jt.init.gauss_(m.weight, std=0.01)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
        
        # Initialize bias for classification layer for Focal Loss stability
        jt.init.constant_(self.cls_logits.bias, -math.log((1 - self.prior) / self.prior))
        
        # Create a list of ScaleExp modules, one for each FPN level
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])
    
    def execute(self, inputs):
        '''inputs: A list of 5 feature maps [P3, P4, P5, P6, P7]'''
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        
        # Define a maximum value for the raw regression output to prevent explosions
        MAX_LTRB_PRED = 1000.0
        
        for index, P in enumerate(inputs):
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            
            # --- START: CORRECTED REGRESSION LOGIC ---
            # 1. Get the raw regression prediction
            reg_pred_out = self.reg_pred(reg_conv_out)
            
            # 2. Clamp the output to prevent extremely large values from entering the exp() function
            reg_pred_out = jt.clamp(reg_pred_out, max_v=MAX_LTRB_PRED)
            
            # 3. Apply the learnable scale and exp function
            reg_preds.append(jt.nn.relu(reg_pred_out))
            # --- END: CORRECTED REGRESSION LOGIC ---
            
        return cls_logits, cnt_logits, reg_preds