# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from block import Conv3Block,Conv2Block

class Unet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.convblock1 = Conv2Block(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 256

        self.convblock2 = Conv2Block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 128

        self.convblock3 = Conv2Block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # 64

        self.convblock4 = Conv2Block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # 64

        self.convblock5= Conv2Block(256, 512)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.convblock6 = Conv2Block(512, 256)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.convblock7 = Conv2Block(256, 128)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.convblock8 = Conv2Block(128, 64)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = x.float()
        conv1 = self.convblock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.convblock2(pool1)
        pool2 = self.pool1(conv2)

        conv3 = self.convblock3(pool2)
        pool3 = self.pool1(conv3)

        conv4 = self.convblock4(pool3)
        pool4 = self.pool1(conv4)

        conv5 = self.convblock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.convblock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.convblock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.convblock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)

        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        conv9 = self.lrelu(self.conv9_3(conv9))
        #print(conv9.shape)

        return conv9

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
