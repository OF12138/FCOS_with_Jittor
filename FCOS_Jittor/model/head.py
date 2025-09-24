import jittor as jt
import jittor.nn as nn
import math
import numpy as np

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
        

        # --- START: ADD THIS BLOCK TO FIX REGRESSION ---
        # Initialize the regression head's bias to a small positive value.
        # This encourages the model to predict non-zero boxes from the start.
        # A bias of log(4) means the initial predicted offsets will be around exp(log(4)) = 4 pixels.
        initial_bias = math.log(4.0)
        jt.init.constant_(self.reg_pred.bias, initial_bias)
        # --- END: ADD THIS BLOCK ---


        # Create a list of ScaleExp modules, one for each FPN level
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)]) # could be exp(1.0)
    
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
            reg_pred_out = jt.clamp(reg_pred_out, max_v=10.0)
            
            # 3. Apply the learnable scale and exp function
            #reg_preds.append(jt.nn.relu(reg_pred_out))
            reg_preds.append(self.scale_exp[index](reg_pred_out))
            # --- END: CORRECTED REGRESSION LOGIC ---
            
        return cls_logits, cnt_logits, reg_preds


#---test---





if __name__ == '__main__':
    # This main block is for testing and comparing against the PyTorch implementation.
    
    # Import PyTorch
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Skipping comparison test.")
        exit()

    # --- START: Self-Contained PyTorch Implementation ---
    class ScaleExpPytorch(torch.nn.Module):
        def __init__(self,init_value=1.0):
            super(ScaleExpPytorch,self).__init__()
            self.scale=torch.nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
        def forward(self,x):
            return torch.exp(x*self.scale)

    class ClsCntRegHeadPytorch(torch.nn.Module):
        def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
            super(ClsCntRegHeadPytorch,self).__init__()
            self.prior=prior
            self.class_num=class_num
            self.cnt_on_reg=cnt_on_reg
            
            cls_branch=[]
            reg_branch=[]

            for i in range(4):
                cls_branch.append(torch.nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
                if GN:
                    cls_branch.append(torch.nn.GroupNorm(32,in_channel))
                cls_branch.append(torch.nn.ReLU(True))

                reg_branch.append(torch.nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
                if GN:
                    reg_branch.append(torch.nn.GroupNorm(32,in_channel))
                reg_branch.append(torch.nn.ReLU(True))

            self.cls_conv=torch.nn.Sequential(*cls_branch)
            self.reg_conv=torch.nn.Sequential(*reg_branch)

            self.cls_logits=torch.nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
            self.cnt_logits=torch.nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
            self.reg_pred=torch.nn.Conv2d(in_channel,4,kernel_size=3,padding=1)
            
            self.apply(self.init_conv_RandomNormal)
            
            torch.nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))
            initial_bias = math.log(4.0)
            torch.nn.init.constant_(self.reg_pred.bias, initial_bias)

            self.scale_exp = torch.nn.ModuleList([ScaleExpPytorch(1.0) for _ in range(5)])
        
        def init_conv_RandomNormal(self,module,std=0.01):
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                     if module not in [self.cls_logits, self.reg_pred]:
                        torch.nn.init.constant_(module.bias, 0)
        
        def forward(self,inputs):
            cls_logits=[]
            cnt_logits=[]
            reg_preds=[]
            for index,P in enumerate(inputs):
                cls_conv_out=self.cls_conv(P)
                reg_conv_out=self.reg_conv(P)
                cls_logits.append(self.cls_logits(cls_conv_out))
                if not self.cnt_on_reg:
                    cnt_logits.append(self.cnt_logits(cls_conv_out))
                else:
                    cnt_logits.append(self.cnt_logits(reg_conv_out))
                reg_pred_out = self.reg_pred(reg_conv_out)
                reg_pred_out = torch.clamp(reg_pred_out, max=10.0)
                reg_preds.append(self.scale_exp[index](reg_pred_out))
            return cls_logits,cnt_logits,reg_preds
    # --- END: Self-Contained PyTorch Implementation ---


    print("--- Starting Jittor vs. PyTorch Head Comparison ---")
    
    # Set Jittor to use GPU if available
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("Jittor is using GPU.")
    else:
        print("Jittor is using CPU.")

    # Set seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    jt.set_global_seed(0)

    # --- 1. Define Model and Input Parameters ---
    in_channels = 256
    num_classes = 80
    batch_size = 2
    fpn_dims = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]

    # --- 2. Create Identical Mock Input Data ---
    print("\nCreating mock FPN input data...")
    mock_fpn_input_np = [np.random.randn(batch_size, in_channels, h, w).astype('float32') for h, w in fpn_dims]
    
    mock_fpn_input_jt = [jt.array(arr) for arr in mock_fpn_input_np]
    mock_fpn_input_pt = [torch.from_numpy(arr) for arr in mock_fpn_input_np]
    print(f"Input generated for {len(mock_fpn_input_np)} FPN levels with batch size {batch_size}.")

    # --- 3. Instantiate Models ---
    print("\nInstantiating Jittor and PyTorch models...")
    jittor_head = ClsCntRegHead(in_channels, num_classes)
    pytorch_head = ClsCntRegHeadPytorch(in_channels, num_classes)
    
    jittor_head.eval()
    pytorch_head.eval()

    # --- 4. Copy Weights from PyTorch to Jittor for a Fair Comparison ---
    print("\nCopying weights from PyTorch model to Jittor model...")
    pt_params = dict(pytorch_head.named_parameters())
    jt_params = dict(jittor_head.named_parameters())

    for name, jt_param in jt_params.items():
        if name in pt_params:
            pt_param = pt_params[name]
            jt_param.assign(pt_param.detach().cpu().numpy())
            # print(f"  - Copied '{name}'")
    print("Weight copy complete.")

    # --- 5. Run Inference ---
    print("\nRunning inference on both models...")
    with jt.no_grad():
        cls_logits_jt, cnt_logits_jt, reg_preds_jt = jittor_head(mock_fpn_input_jt)

    with torch.no_grad():
        cls_logits_pt, cnt_logits_pt, reg_preds_pt = pytorch_head(mock_fpn_input_pt)
    print("Inference complete.")

    # --- 6. Compare Outputs ---
    print("\n--- Comparing Outputs ---")
    all_match = True
    
    def compare_tensors(name, jt_tensor, pt_tensor, atol=1e-5):
        global all_match
        jt_np = jt_tensor.numpy()
        pt_np = pt_tensor.detach().cpu().numpy()
        if np.allclose(jt_np, pt_np, atol=atol):
            print(f"✅ {name}: Outputs MATCH")
        else:
            all_match = False
            diff = np.abs(jt_np - pt_np).max()
            print(f"❌ {name}: Outputs DO NOT MATCH (max difference: {diff})")

    for i in range(len(fpn_dims)):
        print(f"\n--- Level P{i+3} ---")
        compare_tensors(f"Cls Logits [Level {i}]", cls_logits_jt[i], cls_logits_pt[i])
        compare_tensors(f"Cnt Logits [Level {i}]", cnt_logits_jt[i], cnt_logits_pt[i])
        compare_tensors(f"Reg Preds  [Level {i}]", reg_preds_jt[i], reg_preds_pt[i])

    print("\n--- Final Result ---")
    if all_match:
        print("🎉🎉🎉 Congratulations! All Jittor and PyTorch outputs are identical.")
    else:
        print("🔥🔥🔥 Mismatch detected. There is a difference in the implementations.")









