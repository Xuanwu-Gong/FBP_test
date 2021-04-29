# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from block import Conv2Block,Conv3Block

class FBP(nn.Module):
    def __init__(self,out_channels=1):
        super(FBP, self).__init__()
        self.convblock1_1 = Conv3Block(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 256

        self.convblock1_2 = Conv3Block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 128

        self.convblock1_3 = Conv3Block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # 64

        self.conv1_1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        # back projection FC层
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=4096, out_features=4096)

        self.conv1_2 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)  # 64

        # 上采样
        self.upv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 128
        self.convblock2_1 = Conv3Block(64, 64)

        self.upv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 256
        self.convblock2_2 = Conv3Block(32, 32)

        self.upv3 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # 512
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.float()
        conv1 = self.convblock1_1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.convblock1_2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.convblock1_3(pool2)
        pool3 = self.pool3(conv3)
        pool3= self.conv1_1(pool3)

        b,c,h,w=pool3.shape
        pool3=torch.reshape(pool3,(b,c*h*w))
        dropout=self.dropout(pool3)
        fc=self.fc(dropout)
        fc_reshape=torch.reshape(fc,(b,1,64,64))
        fc_reshape=self.conv1_2(fc_reshape)

        up1=self.upv1(fc_reshape)
        up1=self.convblock2_1(up1)

        up2 = self.upv2(up1)
        up2 = self.convblock2_2(up2)

        up3 = self.upv3(up2)
        up3 = self.conv1(up3)
        up3 = self.relu(up3)
        up3 = self.conv2(up3)
        up3 = self.relu(up3)#32 channels
        middle = self.conv3(up3)#1 channels

        return middle
        #return middle, up3

    def _initialize_weights(self):
        classname = self.__class__.__name__
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
                # print('kaiming')

            if classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)  # bn层里初始化γ，服从（1，0.02）的正态分布
                m.bias.data.fill_(0)  # bn层里初始化β，默认为0
