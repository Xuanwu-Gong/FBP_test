import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conv3Block(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        # padding=(kernel_size-1)/2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, **kwargs)
        self.bn1 = nn.BatchNorm2d(output_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, **kwargs)
        self.bn2 = nn.BatchNorm2d(output_channels, affine=True)
        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, **kwargs)
        self.bn3 = nn.BatchNorm2d(output_channels, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

class Conv2Block(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        # padding=(kernel_size-1)/2
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, **kwargs)
        self.bn1 = nn.BatchNorm2d(output_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, **kwargs)
        self.bn2 = nn.BatchNorm2d(output_channels, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x