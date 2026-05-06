import torch
import torch.nn as nn
from Model.Modules import DepthwiseSeparableConv
from Model.CoordAtt import CoordAtt
from Model.SimplifiedSelfAttention import SimplifiedSelfAttention
from Model.GlobalAtt import GlobalAtt
from Model.GlobalAttAll import GlobalAttAll
class RefineFM(nn.Module):
    def __init__(self,  in_channel):
        super(RefineFM, self).__init__()

        self.LA1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channel //3  , out_channels=in_channel // 3, 
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(in_channel //3 ),
            nn.GELU(), 
        )

        self.LA3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channel //3  , out_channels=in_channel // 3, 
                                          kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_channel //3 ),
            nn.GELU(), 
        )
        self.attention = CoordAtt(in_channel // 3)

        self.gfa = GlobalAtt(in_channel // 3)
        # self.gfa = GA(in_channel // 3)
        # self.gfa = SimplifiedSelfAttention(in_channel // 3, 3)
        self.Agfa = GlobalAttAll(in_channel)
        self.caconv3 =  nn.Sequential(
            DepthwiseSeparableConv(in_channel, in_channel, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.GELU(), 
        )
        self.caconv1 =  nn.Sequential(
            DepthwiseSeparableConv(in_channel, in_channel, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(in_channel),
            nn.GELU(), 
        )        

    def forward(self, x):
        shortcut = x.clone()

        dims = x.shape[1] // 3
        xatt = self.Agfa(x)
        x1, x2, x3 = torch.split(x, [dims, dims, dims], dim=1) 

        if x1.shape[2] > 6:
            x1 = self.LA3(x1)
        else:
            x1 = self.LA1(x1)

        x2 = self.gfa(x2)

        x3 = self.attention(x3)

        x_cat = torch.cat((x1, x2, x3), 1)                             

        x_cat = x_cat  + xatt * x_cat
        if x.shape[2] > 6:
            outx = shortcut + self.caconv3(x_cat)
        else:
            outx = shortcut + self.caconv1(x_cat)
        return outx