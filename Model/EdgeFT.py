import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Modules import DepthwiseSeparableConv
class EdgeFT(nn.Module):
    def __init__(self, channel):
        super(EdgeFT, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)    
        self.conv2 =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=12 , out_channels=channel,
                                          kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(channel),
            nn.GELU(),    
        ) 

        self.conv3 =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=24 , out_channels=channel,
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(channel),
            nn.GELU(),      
        ) 
        self.conv4 =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=30 , out_channels=channel,
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(channel),
            nn.GELU(),     
        )  
        self.catfconv33 =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=36 , out_channels=36,
                                          kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(36),
            nn.GELU(),                                                       
        )  
        self.catfconv13 =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=36 , out_channels=36,
                                          kernel_size=(1,3), stride=1,padding=(0,1)),
            nn.BatchNorm2d(36),
            nn.GELU(),                                                       
        )  
        self.catfconv31 =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=36 , out_channels=36,
                                          kernel_size=(3,1), stride=1,padding=(1,0)),
            nn.BatchNorm2d(36),
            nn.GELU(),                                                       
        ) 
        self.catfconv =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=36 , out_channels=channel,
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(channel),
            nn.GELU(),                                                       
        )         
        self.covne =  nn.Sequential(
            DepthwiseSeparableConv(in_channels=12 , out_channels=channel,
                                          kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(channel),
            nn.GELU(),                                                   
        )   
        self.fc = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(True),
            nn.Linear(12, 12),
            nn.Sigmoid(),
        )         
    def forward(self, FE1, FE2, FE3, FE4):
        # torch.Size([32, 12, 48, 48]) torch.Size([32, 12, 24, 24]) torch.Size([32, 24, 12, 12]) torch.Size([32, 30, 6, 6])
        b = FE1.shape[0]
        f1 = self.pool(FE1)
        gap = f1.view(b, -1)
        feat = self.fc(gap)
        gate = feat[:, -1].view(b, 1, 1, 1)
        
        f2 = F.interpolate(FE2, size=FE1.shape[2:], mode='bilinear', align_corners=False)
        f3 = F.interpolate(FE3, size=FE1.shape[2:], mode='bilinear', align_corners=False)
        f4 = F.interpolate(FE4, size=FE1.shape[2:], mode='bilinear', align_corners=False)

        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)
        catf3 = torch.cat([f2, f3, f4], dim=1)
        cat33 = self.catfconv33(catf3)
        cat13 = self.catfconv13(catf3)
        cat31 = self.catfconv31(catf3)
        catf313 = cat33+cat13+cat31

        y = self.catfconv(catf313)

        outy = gate * y
        
        outy = self.covne(outy)
        return outy

