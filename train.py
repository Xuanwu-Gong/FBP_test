# encoding: utf-8
import os, time, sys, scipy.io
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader
from Unet import Unet
from dataset import MyDataset
from utils import show_time, save_checkpoint, batch_psnr, load_checkpoint
import matplotlib.pyplot as plt
from loss import GeneratorLoss
from Dis import Discriminator
import cv2

sys.path.append('D:\Allcode\Python\RadonAnalyze\pytorch_radon')
from pytorch_radon import IRadon

use_checkpoint = True
load_checkpoint_path = './checkpoints_withUnet/checkpoint_4epoch_10.ckpt'
save_checkpoint_path = './checkpoints_withUnet/'

batch_size = 2
num_workers = 4
epoch = 50
init_epoch = 0
check_loss = 100

size = 512  # set size
delt = 0.5  # smallest angle change
degree = 180.  # Constant Scan range from 0 to 180 degree
iradon = IRadon(size, np.arange(0., degree, delt))  # From full degree scan to image area

print('%s  start loading data...' % show_time(datetime.datetime.now()))
input_list = '/home/littlesunchang/NNFBP/lists/radon_org_200-1013_10w_train.txt'
train_data = MyDataset(input_txt=input_list)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # CUDNN optimization
#***********************************************#
iradon = iradon.cuda().to(device)
#***********************************************#
net = Unet().to(device)
net._initialize_weights()
print('# generator parameters:', sum(param.numel() for param in net.parameters()))

netD = Discriminator().to(device)
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()
#optimizer = optim.Adam(net.parameters(), lr=3e-4)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5)
optimizerD = optim.Adam(netD.parameters(), lr=1e-5)

if use_checkpoint:
    net, _, optimizerD, init_epoch= load_checkpoint(net, netD, optimizerD, load_checkpoint_path)
    print('%s checkpoint loaded!!!' % load_checkpoint_path)

save_num=10
save_interval = int(len(train_loader) / save_num)
print('save_interval=', save_interval)

print('%s  Start training...' % show_time(datetime.datetime.now()))
for epoch in range(init_epoch, epoch + 1):
    losses = 0
    d_losses = 0
    count = 0
    step_loss = []
    d_step_loss = []
    start_time = time.time()

    net.train()
    netD.train()

    for step, data in enumerate(train_loader):
        _, input, gt, _, _ = data
        input = input.cuda().to(device)
        gt = gt.cuda().to(device)
        count += len(input)
        net.train()
        middle, ResPrediction = net(input)
        #Initial matrix
        RadonRepair = torch.randn(1, 1, 512, 360)
        RadonRepair = RadonRepair.cuda().to(device)
        #残差和input相加得到的修复后的Radon——1DFFT ,操作在GPU上
        Prediction = input + ResPrediction
        
        ResizePrediction = Prediction[:, :, :, 12:372]
        
        #1D IFFT Transformation
        for i in range(360):
            Real = ResizePrediction[0, 0, :, i]
            print(Real.shape)
            Imag = ResizePrediction[0, 1, :, i]
            compl = torch.stack([Real, Imag],dim=2)
            IFFT1D = torch.ifft(compl)
            RadonRepair[0, 0, :, i] = IFFT1D[:, 0]
            
                    
        PredictionImg = iradon(RadonRepair)
        
        dtype = torch.cuda.FloatTensor
        gt = gt.type(dtype)
        
        
        
        # update D
        netD.zero_grad()
        real_out = netD(gt).mean()
        fake_out = netD(PredictionImg).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        d_losses += d_loss.item()

        # update G
        net.zero_grad()
        loss = generator_criterion(fake_out, prediction, gt, middle)
        loss.backward()
        optimizer.step()
        losses += loss.item()

        present = time.time()
        rest_time = (present - start_time) * (len(train_loader) - (step + 1)) / (step + 1)
        psnr, ssim = batch_psnr(prediction, gt, 255)

        print('epoch%d, step %4d/%4d，剩余时间%dmin, g_loss=%.3f real=%.3f, fake=%.3f, batch_PSNR=%.3f, batch_SSIM=%.3f' % (
            epoch + 1,
            step + 1,
            len(train_loader),
            int(rest_time / 60.0),
            losses / (step + 1),
            real_out,
            fake_out,
            psnr,
            ssim))

        step_loss.append(losses / (step + 1))
        d_step_loss.append(d_losses / (step + 1))
        if ((step + 1) % save_interval == 0):
            part = (step + 1) / save_interval
            if not os.path.exists(save_checkpoint_path):
                os.mkdir(save_checkpoint_path)
            save_checkpoint(net, netD, optimizer, optimizerD, epoch + 1, step_loss, d_losses,
                            save_checkpoint_path + 'checkpoint_%depoch_%d.ckpt' % (epoch + 1, part))

    print('\n%s  epoch %d: Average_loss=%f\n' % (
    show_time(datetime.datetime.now()), epoch + 1, losses / len(train_loader)))
