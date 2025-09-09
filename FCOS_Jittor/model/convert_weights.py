import torch
import jittor as jt
from resnet import resnet50 as resnet50_jittor  # Import your Jittor ResNet50
from torchvision.models import resnet50 as resnet50_torch # Import PyTorch's ResNet50

def convert():
    # --- Load PyTorch Model ---
    # 1. Instantiate the PyTorch model
    torch_model = resnet50_torch()
    # 2. Load the downloaded .pth file
    # Make sure the .pth file is in the same directory or provide the full path
    torch_weights = torch.load("./model/resnet50-19c8e357.pth")
    torch_model.load_state_dict(torch_weights)
    torch_model.eval()
    print("PyTorch model and weights loaded successfully.")

    # --- Load Jittor Model ---
    # 1. Instantiate your Jittor model
    jittor_model = resnet50_jittor()
    jittor_model.eval()
    print("Jittor model instantiated successfully.")

    # --- Convert and Copy Weights ---
    # 2. Copy weights from PyTorch to Jittor
    torch_params = dict(torch_model.named_parameters())
    jittor_params = dict(jittor_model.named_parameters())

    for name, jittor_param in jittor_params.items():
        # Jittor's BatchNorm has 'weight' and 'bias'
        # PyTorch's running_mean and running_var are part of the state_dict but not parameters
        # We need to handle them separately.
        
        if name in torch_params:
            torch_param = torch_params[name]
            # Copy the parameter data
            jittor_param.data = torch_param.detach().cpu().numpy()
            print(f"Copied parameter: {name}")

    # Handle BatchNorm running_mean and running_var
    for torch_name, torch_buf in torch_model.named_buffers():
        if torch_name in jittor_params:
            jittor_params[torch_name].data = torch_buf.detach().cpu().numpy()
            print(f"Copied buffer: {torch_name}")


    # --- Save Jittor Model ---
    # 3. Save the Jittor model's weights to a .pkl file
    jittor_model.save("resnet50_jittor.pkl")
    print("\nConversion complete! Weights saved to resnet50_jittor.pkl")

if __name__ == "__main__":
    # Set Jittor to use the same device as your main project if needed
    jt.flags.use_cuda = 1 # or 0 for CPU
    convert()