import jittor.nn as nn
import jittor as jt
from jittor.nn import init
import math

class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=jt.array([init_value]).float32()
    def execute(self,x):
        return jt.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg
        
        cls_branch=[]
        reg_branch=[]

        for i in range(4):
            cls_branch.append(nn.Conv(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))
            cls_branch.append(nn.ReLU())

            reg_branch.append(nn.Conv(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU())

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)

        self.cls_logits=nn.Conv(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits=nn.Conv(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred=nn.Conv(in_channel,4,kernel_size=3,padding=1)
        
        self.apply(self.init_conv_RandomNormal)
        
        jt.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])
    
    def init_conv_RandomNormal(self, module):
        if isinstance(module, nn.Conv):
            # Using init.gauss_ instead of init.normal_ which is not a top-level function.
            init.gauss_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def execute(self,inputs):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):

            # --- ADD THIS DEBUGGING BLOCK ---
            #print(f"\n--- DEBUG: Inside Head, FPN Level {index+3} ---")
            #print(f"  Input P shape: {P.shape}")
            # --- END OF DEBUGGING BLOCK ---

            cls_conv_out=self.cls_conv(P)
            reg_conv_out=self.reg_conv(P)


            # --- ADD THIS DEBUGGING BLOCK ---
            #print(f"  cls_conv_out shape: {cls_conv_out.shape}")
            #print(f"  reg_conv_out shape: {reg_conv_out.shape}")
            # --- END OF DEBUGGING BLOCK ---

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits,cnt_logits,reg_preds