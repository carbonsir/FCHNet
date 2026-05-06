import torch
import torch.nn as nn
from Model.Modules import ConvBN, DepthwiseSeparableConv
class GlobalAttAll(nn.Module):
    def __init__(self, channels):
        super(GlobalAttAll, self).__init__()

        self.global_attH = nn.Sequential(
            # squeeze channel dimension in forward function
            # squeeze spatial
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
            # excitation spatial
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):

        x = self.global_attH(torch.mean(x, dim=1, keepdim=True)) 
        return x

    