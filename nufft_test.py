# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:32:21 2020

@author: gwx
"""
import math
import pydicom
import cv2
import numpy as np
import os, sys

sys.path.append('D:\Allcode\Python\RadonAnalyze\pytorch_radon')
from pytorch_radon import Radon, IRadon

from skimage.measure import compare_psnr, compare_ssim
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon


import matplotlib.pyplot as plt
import torch


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import scipy.ndimage
import cv2

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
 
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    
    return rotated #7

size = 512  # set size
delt = 1.  # smallest angle change
degree = 180.  # Constant Scan range from 0 to 180 degree
width = 512 #设置傅里叶变换宽度

nnradon = Radon(size, np.arange(0., degree, delt))
nniradon = IRadon(size, np.arange(0., degree, delt))  # From full degree scan to image area

file = '../RadonAnalyze/dcmdata/000056.dcm'
ctfile = pydicom.dcmread(file)
print(ctfile)

prhalf = 2 * np.arange(2/width, width/2, 1,  dtype = np.float32) / width
afhalf = 2 * np.arange(width/2, 2/width, -1,  dtype = np.float32) / width
r_l_filter = np.append(prhalf, afhalf)  #设置滤波器，R-L是一种基础的滤波算法

img = ctfile.pixel_array
img[img == -2000] = 0
img = img.astype('float32')
img = ( (img - img.min()) / (img.max() - img.min()) ) * 255.
# plt.figure(figsize=(8, 5))# figure 1
# plt.imshow(img, cmap=plt.cm.bone)
# plt.show()
# print(img)
theta = np.linspace(0., 180., 180, endpoint=False)
sinogram = radon(img, theta=theta, circle=True)

radonFFT = np.zeros((512,512)).astype(np.complex128)
radon1DFFT = np.zeros((512,sinogram.shape[1])).astype(np.complex128)
rot1D = np.zeros((512,512)).astype(np.complex128)


for i in range(sinogram.shape[1]):
    radon1DFFT[:, i] = np.fft.fft(sinogram[:, i])
    

    radon1DFFT[:,i] = np.multiply(radon1DFFT[:,i], r_l_filter)
    tempFT = radon1DFFT[:, i].copy()
    radon1DFFT[0:256, i] = tempFT[256:512]
    radon1DFFT[256:512, i] = tempFT[0:256]

plt.figure()
plt.plot(abs(radon1DFFT[:,0]))
plt.show()

for i in range(sinogram.shape[1]):
    radon_cache_re = np.zeros((512,512)).astype(np.float64)
    radon_cache_im = np.zeros((512,512)).astype(np.float64)
    radon_cache_re[:,int(np.floor(radon_cache_re.shape[1]/2))] = radon1DFFT[:, i].real
    radon_cache_im[:,int(np.floor(radon_cache_im.shape[1]/2))] = radon1DFFT[:, i].imag
    rot1D_re = rotate(radon_cache_re, i+90)
    rot1D_im = rotate(radon_cache_im, i+90)
    rot1D.real = rot1D_re
    rot1D.imag = rot1D_im
    radonFFT = radonFFT + rot1D
    
plt.figure(figsize=(8, 5))# figure 1
plt.imshow(sinogram, cmap=plt.cm.bone)
plt.show()

print(radonFFT.shape)



# imgreshape = img.reshape(1, 1, 512, 512)
# imgreshape = torch.from_numpy(imgreshape)
# radon_imgshape = nnradon(imgreshape)
# radon_img = np.array(radon_imgshape[0, 0, :, :])
# radonpsnr1 = compare_psnr(sinogram, radon_img, 1e5)
# plt.figure(figsize=(8, 5))# figure 1
# plt.imshow(radon_img, cmap=plt.cm.bone)
# plt.show()
# print(radonpsnr1)

# plt.figure(figsize=(8, 5))# figure 1
# plt.imshow(img, cmap=plt.cm.bone)
# plt.show()


# imgreshape = img.reshape(1, 1, 512, 512)
# imgreshape = torch.from_numpy(imgreshape)
# radon_imgshape = radon(imgreshape)
# radon_img = np.array(radon_imgshape[0, 0, :, :])

# radon_cache = radon_img.copy()

# plt.figure(figsize=(8, 5)) # figure 2
# plt.imshow(radon_img, cmap=plt.cm.bone)
# plt.show()


fft = np.fft.fft2(img)
fft = np.fft.fftshift(fft)
imfft = np.log(1 + abs(fft))
fftlist = np.stack((fft.real, fft.imag))

plt.figure(figsize=(8, 5)) # figure 2
plt.imshow(imfft, cmap=plt.cm.bone)
plt.show()


#radonFFT = radonFFT * (np.abs(radonFFT).max() / np.abs(radonFFT).max())
#radonFFT = np.rot90(radonFFT)
# plt.figure(figsize=(8, 5)) # figure 2
# plt.imshow(fftlist[0, :, :], cmap=plt.cm.bone)
# plt.show()

# plt.figure(figsize=(8, 5)) # figure 2
# plt.imshow(fftlist[1, :, :], cmap=plt.cm.bone)
# plt.show()
FFTImag = np.log(1 + np.abs(radonFFT))
plt.figure(figsize=(8, 5))# figure 1
plt.imshow(FFTImag, cmap=plt.cm.bone)
plt.show()

IradonImg = iradon(sinogram, theta=theta, output_size=512)
plt.figure(figsize=(8, 5))# figure 1
plt.imshow(IradonImg, cmap=plt.cm.bone)
plt.show()

ishift = np.fft.ifftshift(radonFFT)
img_re = np.fft.ifft2(radonFFT)
plt.figure(figsize=(8, 5))# figure 1
plt.imshow(np.log(1 + np.abs(img_re)), cmap=plt.cm.bone)
plt.show()
# imgpsnr1 = compare_psnr(img, IradonImg, 1e5)
# print(imgpsnr1)
