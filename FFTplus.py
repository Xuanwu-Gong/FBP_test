# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 23:12:58 2020

@author: HP
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

size = 512  # set size
delt = 1.  # smallest angle change
degree = 180.  # Constant Scan range from 0 to 180 degree

nnradon = Radon(size, np.arange(0., degree, delt))
nniradon = IRadon(size, np.arange(0., degree, delt))  # From full degree scan to image area

file = '../RadonAnalyze/dcmdata/000056.dcm'
ctfile = pydicom.dcmread(file)
print(ctfile)



img = ctfile.pixel_array
img[img < 5000] = 0
img = img.astype('float32')

img2 = img.copy()
img3 = img.copy()

for i in range(img.shape[0]):
        if(i % 3 == 0):
            img[i, 255] = 0
        else :
            img[i, 255] = 255
            
for j in range(img.shape[1]):
    img[:, j] = img[:, 255]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(i == (511 - j)):
            img2[i, j] = 255
            
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(i == 255):
            img3[i, :] = 255
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if (pow(float(i)-255.5, 2) + pow(float(j)-255.5, 2) <= pow(192,2)):
#             if (abs((float(i)-255.5) / abs(float(j)-255.5)) == 1):
#                 img[i,j] = 255;

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if (pow(float(i)-255.5, 2) + pow(float(j)-255.5, 2) <= pow(192,2)):
#             if ((float(i)-255.5) / abs(float(j)-255.5) == 1):
#                 img2[i,j] = 255;

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if (pow(float(i)-255.5, 2) + pow(float(j)-255.5, 2) <= pow(192,2)):
#             if ((float(i)-255.5) / abs(float(j)-255.5) == -1):
#                 img3[i,j] = 255;

plt.figure(figsize=(8, 5))# figure 1
plt.subplot(1, 3, 1)
plt.imshow(img, cmap=plt.cm.bone)
plt.subplot(1, 3, 2)
plt.imshow(img2, cmap=plt.cm.bone)
plt.subplot(1, 3, 3)
plt.imshow(img3, cmap=plt.cm.bone)
plt.show()
             
# plt.figure(figsize=(8, 5))# figure 1
# plt.imshow(img, cmap=plt.cm.bone)
# plt.show()

# plt.figure(figsize=(8, 5))# figure 1
# plt.imshow(img2, cmap=plt.cm.bone)
# plt.show()

# plt.figure(figsize=(8, 5))# figure 1
# plt.imshow(img3, cmap=plt.cm.bone)
# plt.show()

fft = np.fft.fft2(img)
fft = np.fft.fftshift(fft)
imfft = np.log(1 + abs(fft))
fftlist = np.stack((fft.real, fft.imag))

fft2 = np.fft.fft2(img2)
fft2 = np.fft.fftshift(fft2)
imfft2 = np.log(1 + abs(fft2))


fft3 = np.fft.fft2(img3)
fft3 = np.fft.fftshift(fft3)
imfft3 = np.log(1 + abs(fft3))

plt.figure(figsize=(8, 5))# figure 1
plt.subplot(1, 3, 1)
plt.imshow(imfft, cmap=plt.cm.bone)
plt.subplot(1, 3, 2)
plt.imshow(imfft2, cmap=plt.cm.bone)
plt.subplot(1, 3, 3)
plt.imshow(imfft3, cmap=plt.cm.bone)
plt.show()
# plt.figure(figsize=(8, 5)) # figure 2
# plt.imshow(imfft, cmap=plt.cm.bone)
# plt.show()

# plt.figure(figsize=(8, 5)) # figure 2
# plt.imshow(imfft2, cmap=plt.cm.bone)
# plt.show()

# plt.figure(figsize=(8, 5)) # figure 2
# plt.imshow(imfft3, cmap=plt.cm.bone)
# plt.show()

