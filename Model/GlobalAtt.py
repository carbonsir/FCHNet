import torch
import torch.nn as nn
from Model.Modules import ConvBN, DepthwiseSeparableConv
class GFA(nn.Module):
    def __init__(self, channels):
        super(GFA, self).__init__()

        self.global_attH3 = nn.Sequential(
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

        self.conv1 = ConvBN(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv3 = ConvBN(in_channels=channels, out_channels=channels, kernel_size=3)

    def forward(self, x):
        x = x+ self.global_attH3(torch.mean(x, dim=1, keepdim=True)) * x
        
        if x.shape[2] > 6:
            x = self.conv3(x)
        else:
            x = self.conv1(x)
        return x
    
class GlobalAtt(nn.Module):
    def __init__(self, channel):
        super(GlobalAtt, self).__init__() 

        self.gfa = GFA(channels=channel) 
        self.gelu = nn.GELU() 

    def forward(self, x):
        
        x = self.gelu(self.gfa(x))

        return x