from torch import nn as nn

from torchvision.models import resnet18


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, num_groups=8):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.downsample = downsample
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out


class ResNetFeatureWarpper(nn.Module):
    def __init__(self, shallow_resnet_feature=False):
        super(ResNetFeatureWarpper, self).__init__()

        self.shallow_resnet_feature = shallow_resnet_feature

        resnet = resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        if not shallow_resnet_feature:
            self.layer2 = resnet.layer2

    def forward(self, x):
        out = []
        x = self.conv1(x)
        out.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out.append(x)

        if not self.shallow_resnet_feature:
            x = self.layer2(x)
            out.append(x)

        return out


