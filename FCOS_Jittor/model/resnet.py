import jittor.nn as nn
import jittor as jt
import math

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# IMPORTANT: These URLs are for PyTorch pretrained models.
# Jittor cannot directly load PyTorch .pth files.
# You will need to download these files, convert them to Jittor's format (.pkl or .bin),
# and then update these URLs (or the loading mechanism) to point to your converted Jittor files.
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.Relu()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # ResNet-B
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.Relu()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, if_include_top=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3,
                             bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.Relu()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Assuming AvgPool takes kernel size as first arg for square pool, stride=1 is default
        self.avgpool = nn.Pool(7, stride=1, op='mean')
        if if_include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.if_include_top = if_include_top

        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out3 = self.layer2(x)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        if self.if_include_top:
            x = self.avgpool(out5)
            x = x.reshape(x.size(0), -1) # Use x.shape[0] for batch size
            x = self.fc(x)
            return x
        else:
            return (out3, out4, out5)
    
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm):
                layer.eval()
                # Explicitly stop gradient for BatchNorm parameters when freezing BN
                if layer.weight is not None:
                    layer.weight.stop_grad()
                if layer.bias is not None:
                    layer.bias.stop_grad()
    
    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                m.stop_grad() # This correctly stops gradients for all parameters in these modules
        for i in range(1, stage + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval() # For BatchNorm layers within the stage
            layer.stop_grad() # Stop gradients for all parameters in the entire stage module
            
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print("WARNING: Jittor cannot directly load PyTorch .pth files. Please convert 'resnet18-5c106cde.pth' to a Jittor-compatible format and update the loading path.")
        # Example of how you would load a Jittor-converted model:
        # model.load('path/to/your/converted_resnet18.pkl')
        pass # Placeholder for actual Jittor loading
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("WARNING: Jittor cannot directly load PyTorch .pth files. Please convert 'resnet34-333f7ec4.pth' to a Jittor-compatible format and update the loading path.")
        # model.load('path/to/your/converted_resnet34.pkl')
        pass
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #print("WARNING: Jittor cannot directly load PyTorch .pth files for ResNet50. If you intend to use pretrained weights, please convert 'resnet50-19c8e357.pth' to a Jittor-compatible format and update the loading path.")
        model.load("resnet50_jittor.pkl")
         # Original code had 'pass', retaining for consistency but adding warning.
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("WARNING: Jittor cannot directly load PyTorch .pth files. Please convert 'resnet101-5d3b4d8f.pth' to a Jittor-compatible format and update the loading path.")
        # model.load('path/to/your/converted_resnet101.pkl')
        pass
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        print("WARNING: Jittor cannot directly load PyTorch .pth files. Please convert 'resnet152-b121ed2d.pth' to a Jittor-compatible format and update the loading path.")
        # model.load('path/to/your/converted_resnet152.pkl')
        pass
    return model