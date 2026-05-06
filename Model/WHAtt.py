import torch.nn as nn
from Model.Modules import  DepthwiseSeparableConv 
from Model.CBAM import CBAM

class WHAtt(nn.Module):
    def __init__(self, channel):
        super(WHAtt, self).__init__()
        self.direction_fusion1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )
        self.direction_fusion3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )        
        self.cbma = CBAM(channel)
    def forward(self, xh):
        if xh.shape[2] < 12:
            hf = self.direction_fusion1(xh)
        else:
            hf = self.direction_fusion3(xh)

        hf = self.cbma(hf)
        return hf