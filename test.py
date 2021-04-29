# encoding: utf-8
import os,time,scipy.io
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader
from model import NNFBP, FBP
from dataset import MyDataset
from skimage.measure import compare_psnr, compare_ssim
import skimage
import scipy.misc
from utils import load_checkpoint
import cv2
from PIL import Image


def batch_psnr(img, imgclean, data_range):
    psnr = 0
    for i in range(img.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :], img[i, :, :], data_range=data_range)
    return psnr


def batch_ssim(img, imgclean, data_range):
    ssim = 0
    for i in range(img.shape[0]):
        ssim += compare_ssim(imgclean[i, :, :], img[i, :, :], data_range=data_range)
    return ssim

def test(test_epoch,test_part):
    model_dir = './checkpoints/checkpoint_%depoch_%d.ckpt'%(test_epoch,test_part)
    test_data = MyDataset(input_txt='/home/littlesunchang/NNFBP/lists/radon_org_1-100_100_test.txt')
    #print()

    save_dir_root='./test/'
    if not os.path.exists(save_dir_root):
        os.mkdir(save_dir_root)

    save_dir_root='./test/epoch_%d_%d/'%(test_epoch,test_part)
    if not os.path.exists(save_dir_root):
        os.mkdir(save_dir_root)

    #print(len(test_data))

    batch_size=2
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NNFBP().to(device)
    #model = FBP().to(device)
    #load_checkpoint(model,optim,model_dir)

    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['net_state_dict'],strict=False)
    print(model_dir+' loaded!')

    total_psnr=0
    total_ssim=0

    with torch.no_grad():
        for step, data in enumerate(test_loader):
            input, gt, input_path = data
            input = input.cuda().to(device)

            model.eval()

            prediction = model(input)
            prediction = prediction.cpu().numpy().astype(np.float32).reshape(prediction.shape[0], prediction.shape[2],prediction.shape[3])
            gt = gt.data.cpu().numpy().astype(np.float32).reshape(gt.shape[0], gt.shape[2], gt.shape[3])

            psnr = batch_psnr(prediction, gt, data_range=255)
            ssim = batch_ssim(prediction, gt, data_range=255)
            total_psnr += psnr
            total_ssim += ssim

            for i in range(0, prediction.shape[0]):
                patient = input_path[i].split('/')[-2]
                num = input_path[i].split('/')[-1].split('.')[0]
                save_dir = save_dir_root + patient + '-'
                cv2.imwrite(save_dir + num + '.png', prediction[i])

                print('count=%3d, patient=%s, num=%3s, PSNR=%.3f, SSIM=%.3f' % ((step + 1), patient,num,psnr / gt.shape[0], ssim / gt.shape[0]))

        f = open('./test_log.txt', 'a')
        str='%s, 平均PSNR=%.3f, 平均SSIM=%.3f' % (model_dir, total_psnr / len(test_data), total_ssim / len(test_data))
        f.write(str + '\n')
        f.close()
        print(str)

if __name__=='__main__':
    #for epoch in range(5,7):
    #    for part in range(1,11):
    #        test(epoch,part)
    test(3, 10)
    #for part in range(1, 11):
     #   test(4,part)
