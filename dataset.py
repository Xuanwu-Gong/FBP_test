# encoding: utf-8
# https://blog.csdn.net/sinat_42239797/article/details/90641659
#import torch.nn.functional as F
#import torch
#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#import torchvision.models as models
#from torchvision import transforms, utils
from torch.utils.data import Dataset
#from PIL import Image
import numpy as np
#import torch.optim as optim
#import os
from torch.utils.data import DataLoader
import cv2

# 定义读取文件的格式 读取txt格式的拉登域图像数据,进行一维傅里叶变换并将type = complex数组转化为type = float32数组
def txt_loader(path):
    a = np.loadtxt(path)
    a = a.astype("float32")
    #a = cv2.resize(a, (512, 512))
    b = np.stack((a, a.copy()))
    for ind in range(a.shape[1]):
        c = a[:, ind]
        d = np.fft.fft(c)
        b[0, :, ind] = d.real
        b[1, :, ind] = d.imag
        b = b.astype("float32")
    return b

# 读取图像png格式的原图、进行二维傅里叶变换并将type = complex数组转化为type = float32数组
def png_loader(path):
    a = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    a = a.astype("float32")
    b = np.fft.fft2(a)
    b = np.fft.fftshift(b)
    c = np.stack((b.real, b.imag)).astype('float32')
    return c

class MyDataset(Dataset):
    def __init__(self, input_txt=None,transform=transforms.ToTensor(), txt_loader=txt_loader, png_loader=png_loader):
        super(MyDataset, self).__init__()

        f_input = open(input_txt, 'r')
        input = []
        for line in f_input:
            line = line.strip('\n')
            input.append(line)
        self.input = input
        self.transform = transform
        self.txt_loader = txt_loader
        self.png_loader=png_loader

    def __getitem__(self, index):
        input_path = self.input[index]
        radon_data = self.txt_loader(input_path)
        #radon_data = padding_radon(radon_data)
        #radon_data = radon_data.reshape(radon_data.shape[0], radon_data.shape[1], radon_data.shape[2])
        try:
            #gt_path = input_path.replace('radon', 'image')
            #gt_path = gt_path.replace('txt', 'png')
            split=input_path.split('/')
            case=split[-2]
            num=split[-1].split('.')[0]

            gt_path='/data/dengken/LIDC/image_ct_data/'+case+'/'+num+'.png'
            gt_image = self.png_loader(gt_path)
            #gt_image = gt_image.reshape(gt_image.shape[0], gt_image.shape[1], gt_image.shape[2])
            #print(gt_image.shape)
            return radon_data, gt_image, input_path

        except Exception as e:
            print(gt_path)
            print(e)

    #使用__len__()初始化一些需要传入的参数及数据集的调用
    def __len__(self):
        return len(self.input)

if __name__ == '__main__':
    input_list='/home/littlesunchang/NNFBP/lists/radon_org_100-1013_1w_train.txt'
    test_data = MyDataset(input_txt=input_list)
    train_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=1)
    for step, data in enumerate(train_loader):
        input_merge_radon, gt_radon, input_path = data
        if step<1:
            print(input_merge_radon.shape)
            print(gt_radon.shape)
            input=input_merge_radon.numpy().reshape(input_merge_radon.shape[2],input_merge_radon.shape[3])
            gt = gt_radon.numpy().reshape(gt_radon.shape[2], gt_radon.shape[3])
            cv2.imwrite('input.png', input)
            cv2.imwrite('gt.png', gt)