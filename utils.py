import datetime
import torch
import matplotlib.pyplot as plt
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
import numpy as np
from collections import OrderedDict

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s

def save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, g_losses, d_losses, savepath):
    save_json = {
        'net_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizer_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'epoch': epoch,
        'losses': g_losses,
        'd_losses': d_losses
    }
    torch.save(save_json, savepath)

def load_checkpoint (netG, netD, optimizerD, checkpoint_path):
    pre_load = torch.load(checkpoint_path, map_location='cuda:0')

    #pre_load['net_state_dict'].pop('fbp.conv3.weight')
    #pre_load['net_state_dict'].pop('fbp.conv3.bias')

    #pre_load['net_state_dict'].pop('unet.conv9_2.weight')
    #pre_load['net_state_dict'].pop('unet.conv9_2.bias')

    #pre_load['net_state_dict'].pop('unet.conv9_3.weight')
    #pre_load['net_state_dict'].pop('unet.conv9_3.bias')

    netG.load_state_dict(pre_load['net_state_dict'], strict = False)
    netD.load_state_dict(pre_load['netD_state_dict'], strict = False)
    optimizerD.load_state_dict(pre_load['optimizerD_state_dict'])
    start_epoch = pre_load['epoch']
    return netG, netD, optimizerD, start_epoch

def batch_psnr(img, imclean, data_range):
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    ssim=0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, 0, :, :], img_cpu[i, 0, :, :], data_range=data_range)
        ssim += compare_ssim(imgclean[i, 0, :, :], img_cpu[i, 0, :, :], data_range=data_range)
    return psnr/img_cpu.shape[0], ssim/img_cpu.shape[0]

def draw_loss(g_losses,epoch):
    plotx = range(len(g_losses))
    plt.plot(plotx, g_losses)
    plt.xlabel('step')
    plt.ylabel('training loss')
    fig = plt.gcf()
    fig.savefig('loss_%depoch.jpg'%epoch, dpi=100)
    #plt.save