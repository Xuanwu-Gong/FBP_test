import torch
from torch import nn
from torchvision.models.vgg import vgg16
import pprint

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        #self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        #self.tv_loss = TVLoss()
        #self.L1_loss = nn.L1Loss(reduce=True, size_average=True)
        self.smooth_L1_loss = torch.nn.SmoothL1Loss(reduce=True, size_average=True)

    def forward(self, out_labels, out_images, target_images, middle):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        # Image Loss
        image_loss = self.smooth_L1_loss(out_images, target_images)

        middle_loss = self.smooth_L1_loss(middle, target_images)
        # TV Loss
        #tv_loss = self.tv_loss(out_images)

        #loss=image_loss + 0.001 * adversarial_loss +  2e-8 * tv_loss
        loss = image_loss + 0.001 * adversarial_loss + 0.01 * middle_loss
        #loss = image_loss + 2e-8 * tv_loss

        return loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
