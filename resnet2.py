'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference: 
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import deform_conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


"""
Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

in_channels (int) – Number of channels in the input image

out_channels (int) – Number of channels produced by the convolution

"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = deform_conv.DeformConv2D(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        #print("shortcut")
        if stride != 1 or in_planes != self.expansion*planes:
            #print("stride: ", stride, "    in_planes: ", in_planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #print("&&&&&&&&&    x: ", x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #print('************OUT BEFORE: ', out.size())
        #print('**************X: ', x.size())
        #print('**************X inside: ', x[1])
        #out = out * 2
        #print("self shortcut: ", self.shortcut(x))
        out += self.shortcut(x)
        #print('************OUT AFTER: ', out.size())
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        print("************First Conv**************")
        out = F.relu(self.bn1(self.conv1(x)))
        print("************Layer1 Resnet**************")
        out = self.layer1(out)
        print("************Layer2 Resnet**************")
        out = self.layer2(out)
        print("************Layer3 Resnet**************")
        out = self.layer3(out)
        print("************Layer4 Resnet**************")
        out = self.layer4(out)
        print("************Fully Connected Resnet***************")
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    print("Using Model Resnet18");
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    print("Using Model Resnet34");
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    print("Using Model Resnet50");
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    print("Using Model Resnet101");
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    #net = ResNet18()
    net = ResNet50()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

#test()
