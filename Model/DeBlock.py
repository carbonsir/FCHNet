import torch.nn as nn
import torch.nn.functional as F
from Model.Modules import DepthwiseSeparableConv
from utils.dctNew import idct_2d
class DeBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeBlock, self).__init__()

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channel, out_channels=out_channel, kernel_size=1, 
                                              padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel, kernel_size=1, 
                                              padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel, kernel_size=(1, 3),
                                              padding=(0, 1), bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )                                              
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 1),
                                              padding=(1, 0), bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),                                              
        )
        self.conve = nn.Sequential(
            DepthwiseSeparableConv(in_channels=12, out_channels=out_channel, kernel_size=1,
                                              padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),                                              
        )

        self.conve1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel, kernel_size=1, 
                                              padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),                                              
        )                                              
        self.conve2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel, kernel_size=(1, 3),
                                              padding=(0, 1), bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),                                              
        )                                              
        self.conve3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 1),
                                              padding=(1, 0), bias=True),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),                                              
        )         
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self, x, e):
        x = idct_2d(x)
        x = self.conv(x)

        x1 = self.conv1(x) 
        x3 = self.conv2(x) 
        x5 = self.conv3(x) 
        
        e = F.interpolate(e, size=x.shape[2:], mode='bilinear', align_corners=False)
        e= self.conve(e)
        e1 = self.conve1(e)
        e3 = self.conve2(e)
        e5 = self.conve3(e) 

        x = x1 + e1 + x3 + e3 + x5 + e5

        x = self.bn(x)
        return x   