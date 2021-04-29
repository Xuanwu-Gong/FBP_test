# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Unet import Unet
from fbp import FBP
from block import Conv2Block,Conv3Block

class NNFBP (nn.Module):
    def __init__(self, num_classes=10):
        super(NNFBP, self).__init__()
        self.FBP = FBP(out_channels=1)

        for para in self.FBP.parameters():
            para.requires_grad = True

        #self.unet = Unet(input_channels=32)
        #self.conv1 = nn.Conv2d(32,8,kernel_size=3,padding=1,stride=1)

        #self.relu = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(8,1,kernel_size=3,padding=1,stride=1)

    def forward(self, x):
        x = x.float()
        #middle, fbp32 = self.FBP(x)
        middle=self.FBP(x)
        #unet = self.unet(fbp32)
        #out = fbp32 + unet
        #out = self.conv1(out)
        #out = self.relu(out)
        #out = self.conv2(out)

        #return middle, out
        return middle

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



if __name__ == '__main__':
    net = NNFBP()
    #dis = Discriminator()
    print('total params =',sum(param.numel() for param in net.parameters()))
    #print('total params =', sum(param.numel() for param in dis.parameters()))
    data_input = Variable(torch.randn([1, 1, 512, 512]))
    out = net(data_input)
    #out_dis=dis(data_input)

