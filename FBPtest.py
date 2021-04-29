# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:19:20 2020

@author: HP
"""
import math
import pydicom
import cv2
import numpy as np
import os, sys

sys.path.append('D:\GongWuxuan\Code\Python\001_FBP\pytorch_radon')
from pytorch_radon import Radon, IRadon

from skimage.measure import compare_psnr, compare_ssim
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon


import matplotlib.pyplot as plt
import torch


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import scipy.ndimage

file = './data/000056.dcm'
ctfile = pydicom.dcmread(file)
print(ctfile)

width = 512 #设置傅里叶变换宽度

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

prhalf = 2 * np.arange(0, width/2, 1,  dtype = np.float32) / width
afhalf = 2 * np.arange(width/2, 0, -1,  dtype = np.float32) / width
r_l_filter = np.append(prhalf, afhalf)  #设置滤波器，R-L是一种基础的滤波算法

img = ctfile.pixel_array
#img[img == -2000] = 0
img = (img - img.min()) / (img.max() - img.min())
img = img.astype('float32')

thetalist = np.linspace(0., 180., 180, endpoint=False)
sinogram = radon(img, theta=thetalist, circle=True)

proj_filtered = np.zeros((width, 180), dtype=np.complex128)
proj_fft = np.zeros((width, 180), dtype=np.complex128)

#1-D FFT 变换
for i in range(sinogram.shape[1]):
    proj_fft[:, i] = np.fft.fft(sinogram[:, i])
    #tempFT = proj_fft[:, i].copy()
    #proj_fft[0:256, i] = tempFT[256:512]
    #proj_fft[256:512, i] = tempFT[0:256]
    
#滤波 proj_filtered

for j in range(sinogram.shape[1]):
    proj_filtered[:,j] = np.multiply(proj_fft[:,j], r_l_filter)
    #proj_filtered[:,j] = proj_fft[:,j]

fbpSingleImg = np.zeros((1536,1536)).astype(np.float32)
fbpImg = np.zeros((512,512)).astype(np.float32)
fftimg = np.zeros((512,512)).astype(np.complex128)
singlefft = np.zeros((512,512)).astype(np.complex128)
rotsinglefft = np.zeros((512,512)).astype(np.complex128)

for k in range(sinogram.shape[1]):
    oneradon = np.real(np.fft.ifft(proj_filtered[:, k]))
    #oneradon = np.real(np.fft.ifft(proj_fft[:, k]))
    for l in range(fbpSingleImg.shape[1]):
        fbpSingleImg[0 : 512, l] = oneradon
        fbpSingleImg[512 : 1024, l] = oneradon
        fbpSingleImg[1024 : 1536, l] = oneradon
    
    singlefft = np.fft.fft2(fbpSingleImg[512:1024, 512:1024])
    singlefft = np.fft.fftshift(singlefft)
    
    rotFBPSingleImg = rotate(fbpSingleImg, k)
    rotsinglefft = np.fft.fft2(rotFBPSingleImg[512:1024, 512:1024])
    rotsinglefft = np.fft.fftshift(rotsinglefft)
    
    if(k==30):
        plt.figure(figsize=(8, 5))# figure 1
        plt.subplot(1, 2, 1)
        plt.imshow(fbpSingleImg[512:1024, 512:1024], cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
        plt.imshow(rotFBPSingleImg[512:1024, 512:1024], cmap=plt.cm.bone)
        plt.show()

        plt.figure(figsize=(8, 5))# figure 2
        plt.subplot(1, 2, 1)
        plt.imshow(np.log(1 + np.abs(singlefft)), cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
        plt.imshow(np.log(1 + np.abs(rotsinglefft)), cmap=plt.cm.bone)
        plt.show()
        
    fbpImg = fbpImg + rotFBPSingleImg[512:1024, 512:1024]
    fftimg = fftimg + rotsinglefft
    
fbpImg = (fbpImg -fbpImg.min()) / (fbpImg.max() - fbpImg.min())
fbpImg = rotate(fbpImg, 90)
# for i in range(fbpImg.shape[0]):
#     for j in range(fbpImg.shape[1]):
#         if((pow(abs(float(i)-256), 2) + pow(abs(float(j)-256), 2)) >= (pow(float(fbpImg.shape[0]) / 2, 2))):
#             fbpImg[i, j] = fbpImg.min()
            
plt.figure(figsize=(8, 5))# figure 3
plt.subplot(1, 2, 1)
plt.imshow(np.log(1 + np.abs(proj_fft)), cmap=plt.cm.bone)
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + np.abs(proj_filtered)), cmap=plt.cm.bone)
plt.show()

proj_ifft = np.real(np.fft.ifft(proj_filtered.T).T);
plt.figure(figsize=(8, 5))# figure 4
plt.subplot(1, 2, 1)
plt.imshow(sinogram, cmap=plt.cm.bone)
plt.subplot(1, 2, 2)
plt.imshow(proj_ifft, cmap=plt.cm.bone)
plt.show()

plt.figure(figsize=(8, 5)) # figure 5
plt.imshow(fbpImg, cmap=plt.cm.bone)
plt.show()

plt.figure(figsize=(8, 5)) # figure 6
plt.imshow(np.log(1 + np.abs(fftimg)), cmap=plt.cm.bone)
plt.show()

