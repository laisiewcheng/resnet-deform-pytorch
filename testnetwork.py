# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:05:29 2020

@author: User
"""

import os
import torch
from torch import nn
import torch.nn.functional as F

import deform_conv

f = open("model.txt", "a")

class PlainNet(nn.Module):
    def __init__(self):
        print("*********PlainNet**********")
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # convs
        print("%%%%%%%%%%%%%%%%%  FORWARD PASS  %%%%%%%%%%%%%%%%%%")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", file = f)
        print("x1: ", x.size(), file = f)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        print("x2: ", x.size(), file = f)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        print("x3: ", x.size(), file = f)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        print("x4: ", x.size(), file = f)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        print("x5: ", x.size(), file = f)
        x = F.avg_pool2d(x, 28)
        print("x6: ", x.size(), file = f)     
        x = x.view(x.size(0), -1)
        print("x7: ", x.size(), file = f)
       
        x = self.classifier(x)
        print("x8: ", x.size(), file = f)
        
        return F.log_softmax(x, dim=1)


class DeformNet(nn.Module):
    def __init__(self):
        
        #print("DeformNet")
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(128)

        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1) 
        self.conv4 = deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        #self.classifier = nn.Linear(128, 10)
        self.classifier = nn.Linear(128, 10)


    def forward(self, x):
        # convs
        #print("forward pass")
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # deformable convolution
        #print("****************DEFORMABLE CONVOLUTION START*********************")
        # deformable convolution
        offsets = self.offsets(x) # will use the nn.Conv2D in line 69 to generate the offsets
        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)
        #print("&&&&&&&&&&&&&&&DEFORMABLE CONVOLUTION END&&&&&&&&&&&&&&&&&&&&&&&")

#        x = F.avg_pool2d(x, 28)
#        x = x.view(x.size(0), -1)
#        x = self.classifier(x)
#        
#        return F.log_softmax(x, dim=1)
    
        x = F.avg_pool2d(x, 28)
        #print("x2: ", x.size())
        x = x.view(x.size(0), -1)
       
        #x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        #print("x3: ", x.size())
        x = self.classifier(x)
        #print("x4: ", x.size())
        return F.log_softmax(x, dim=1)



        #x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        #x = self.classifier(x)

        #return F.log_softmax(x, dim=1)
        
#print("running deformnet") 
#net = DeformNet()
#y = net(torch.randn(1,3,32,32))
#print(y.size())       


i = 0

class DeformNet2(nn.Module):
    def __init__(self):
      
        super(DeformNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = deform_conv.DeformConv2D(32, 32, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(64)

        #self.classifier = nn.Linear(128, 10)
        self.classifier = nn.Linear(64, 10)


    def forward(self, x):
        # convs
        
        print("forward pass DeformNet2")
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
         # deformable convolution
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
       
        x = F.avg_pool2d(x, 28)
        #print("x2: ", x.size())
        x = x.view(x.size(0), -1)
       
        #x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        #print("x3: ", x.size())
        x = self.classifier(x)
        #print("x4: ", x.size())
        return F.log_softmax(x, dim=1)




def test():
    #net = ResNet18()
    net = DeformNet2()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    print(net)

#test()





















