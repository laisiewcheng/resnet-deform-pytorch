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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, bias=False)
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




'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        #self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        #self.conv3 = deform_conv.DeformConv2D(planes, self.expansion*planes, kernel_size=1, bias=False)
        #self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1) 
        #self.conv3 = deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        #print("shortcut")
        if stride != 1 or in_planes != self.expansion*planes:
            #print("stride: ", stride, "    in_planes: ", in_planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes*2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #print("&&&&&&&&&    x: ", x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        
        #offsets = self.offsets(out)
        #out = F.relu(self.bn2(self.conv2(out, offsets)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))
        #print('************OUT BEFORE: ', out.size())
        #print('**************X: ', x.size())
        #print('**************X inside: ', x[1])
        #out = out * 2
        #print("self shortcut: ", self.shortcut(x))
        #print("out shape: ", out.shape)
        print("x size: ", x.size())
        print("out size: ", out.size())
        out += self.shortcut(x)
        #print('************OUT AFTER: ', out.size())
        out = F.relu(out)
        return out

'''


#Use normal convolution
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        #print("Bottleneck")
        super(Bottleneck, self).__init__()
        
        #cannot apply deformable conv to conv1 and conv3 becasue the kernel size is 1
        
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        #self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1, stride=stride) 
        #self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        
        
        
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
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = F.relu(self.bn2(self.conv2(out)))
        
        #enable this if used deformable conv in conv2
        '''
        offsets = self.offsets(out)
        out = self.conv2(out, offsets)
        out = self.bn2(out)
        out = F.relu(out)
        '''
        out = self.bn3(self.conv3(out))
        
        
        out += self.shortcut(x)
        
        out = F.relu(out)
        
        return out
        
        
        
#Use deformable convolution in 3 x 3 filter       
class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        #print("Bottleneck2")
        super(Bottleneck2, self).__init__()
        
        #cannot apply deformable conv to conv1 and conv3 becasue the kernel size is 1
        
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  
        
        self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1, stride=stride) 
        self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, padding=1)
         
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        
        
        
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
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        #out = F.relu(self.bn2(self.conv2(out)))
        
        #enable this if used deformable conv in conv2
        
        offsets = self.offsets(out)
        out = self.conv2(out, offsets)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.bn3(self.conv3(out))
        
        
        out += self.shortcut(x)
        
        out = F.relu(out)
        
        return out       
         
        
        
        
        

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        #self.layer1 = self._make_layer(Bottleneck2, 64, num_blocks[0], stride=1)
        
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer2 = self._make_layer(Bottleneck2, 128, num_blocks[1], stride=2)
        
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
    
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
       
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
    #net = ResNet50()
    net = ResNet101()
    print("\nResNet101 Architecture\n", net)
    y = net(torch.randn(1,3,32,32))
    print(y.size())


#test()
