import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.CFA import CFA
from Model.EfficientNet import EfficientNet_B0
from Model.EfficientNetV2 import EfficientNetV2
from Model.TinyNet import TinyNetA
from Model.Modules import ConvBNGeLU, DepthwiseSeparableConv
from Model.DeBlock import DeBlock
from Model.EdgeFT import EdgeFT
from Model.HLTBlock import DWT, IDWT
from Model.WHAtt import WHAtt
from Model.WLAtt import WLAtt
from Model.Shufflenet import Shufflenet

class WFR(nn.Module):
    def __init__(self, in_channels, high_channels, prev_channels=None):
        super(WFR, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.low_att = WLAtt(in_channels)
        self.high_att = WHAtt(high_channels)
        if prev_channels is not None:
            self.prev_proj = nn.Sequential(
                DepthwiseSeparableConv(prev_channels, in_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_channels),
                nn.GELU()
            )
        else:
            self.prev_proj = None

    def forward(self, x, prev=None):
        if prev is not None:
            prev = F.interpolate(prev, size=x.shape[2:], mode='bilinear', align_corners=False)
            x_in = x + self.prev_proj(prev)
        else:
            x_in = x

        xdL, xdH = self.dwt(x_in)
        return x + self.idwt(self.low_att(xdL), self.high_att(xdH))

class FCHNet(nn.Module):
    def __init__(self, backbone='efficientb0', Lchannels=(12,24,48,64), Hchannels=(36,72,144,192)):
        super(FCHNet, self).__init__()
        if backbone == 'efficientb0':
            self.encoder_backbone = EfficientNet_B0()
        elif backbone == 'tinynet-a':
            self.encoder_backbone = TinyNetA()         
        else:
            print('backbone error')
            return

        self.bname = backbone
        stage_channels = self.encoder_backbone.get_stage_channels()
        # reduction
        self.re_conv1 = ConvBNGeLU(in_channels=stage_channels[0], out_channels=Lchannels[0], kernel_size=1)
        self.re_conv2 = ConvBNGeLU(in_channels=stage_channels[1], out_channels=Lchannels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(in_channels=stage_channels[2], out_channels=Lchannels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(in_channels=stage_channels[3], out_channels=Lchannels[3], kernel_size=1)

        self.CFA1 = CFA(Lchannels[0], Hchannels[0], Fintervals=(96, Lchannels[0] // 2))
        self.CFA2 = CFA(Lchannels[1], Hchannels[1], Fintervals=(96, Lchannels[1] // 2))
        self.CFA3 = CFA(Lchannels[2], Hchannels[2], Fintervals=(96, Lchannels[2] // 2))
        self.CFA4 = CFA(Lchannels[3], Hchannels[3], Fintervals=(96, Lchannels[3] // 2))

        # activation
        self.gelu = nn.GELU()
        self.BGD4 = DeBlock(Lchannels[3], Lchannels[3])
        self.BGD3 = DeBlock(Lchannels[3], Lchannels[2])        
        self.BGD2 = DeBlock(Lchannels[2], Lchannels[1])
        self.BGD1 = DeBlock(Lchannels[1], Lchannels[0])
       
        # out conv
        self.out_conv1 = nn.Conv2d(Lchannels[0], 1, kernel_size=1, padding=0)
        self.out_conv2 = nn.Conv2d(Lchannels[1], 1, kernel_size=1, padding=0)
        self.out_conv3 = nn.Conv2d(Lchannels[2], 1, kernel_size=1, padding=0)
        self.out_conv4 = nn.Conv2d(Lchannels[3], 1, kernel_size=1, padding=0)
        self.WFR4 = WFR(Lchannels[3], Hchannels[3])
        self.WFR3 = WFR(Lchannels[2], Hchannels[2], prev_channels=Lchannels[3])
        self.WFR2 = WFR(Lchannels[1], Hchannels[1], prev_channels=Lchannels[2])
        self.WFR1 = WFR(Lchannels[0], Hchannels[0], prev_channels=Lchannels[1])
        self.BFE = EdgeFT(Lchannels[0])
        self.linearimg = nn.Conv2d(Lchannels[0], 1, kernel_size=1, stride=1, padding=0)
        self.dwt = DWT()
    def forward(self, x, high, low):   
        if self.bname == "pvtv2":
            endpoints = self.encoder_backbone.extract_endpoints(x)
            x1 = endpoints['reduction_2'] #torch.Size([32, 32, 96, 96]) 
            x2 = endpoints['reduction_3'] #torch.Size([32, 64, 48, 48]) 
            x3 = endpoints['reduction_4'] #torch.Size([32, 160, 24, 24])
            x4 = endpoints['reduction_5'] #torch.Size([32, 256, 12, 12])           
        else:
            x1, x2, x3, x4 = self.encoder_backbone(x)
         
        x1 = self.re_conv1(x1)# torch.Size([32, 12, 96, 96])
        x2 = self.re_conv2(x2)# torch.Size([32, 24, 48, 48])
        x3 = self.re_conv3(x3)# torch.Size([32, 48, 24, 24])
        x4 = self.re_conv4(x4)# torch.Size([32, 64, 12, 12])

        wx4 = self.WFR4(x4)
        wx3 = self.WFR3(x3, wx4)
        wx2 = self.WFR2(x2, wx3)
        wx1 = self.WFR1(x1, wx2)

        new_wxl1, new_wxh1 = self.dwt(wx1)# torch.Size([32, 12, 48, 48]) torch.Size([32, 36, 48, 48] 
        new_wxl2, new_wxh2 = self.dwt(wx2)# torch.Size([32, 24, 24, 24]) torch.Size([32, 72, 24, 24])
        new_wxl3, new_wxh3 = self.dwt(wx3)# torch.Size([32, 48, 12, 12]) torch.Size([32, 144, 12, 12])
        new_wxl4, new_wxh4 = self.dwt(wx4)# torch.Size([32, 64, 6, 6]) torch.Size([32, 192, 6, 6])
       
        y1, edge = self.CFA1(new_wxh1, new_wxl1, high, low)
        y2, FH2 = self.CFA2(new_wxh2, new_wxl2, high, low)
        y3, FH3 = self.CFA3(new_wxh3, new_wxl3, high, low)
        y4, FH4 = self.CFA4(new_wxh4, new_wxl4, high, low)
        
        edgef = self.BFE(edge, FH2, FH3, FH4)
        
        out4 = self.gelu(self.BGD4(y4, edgef))
        out3 = self.gelu(
            self.BGD3(F.interpolate(out4, size=y3.shape[2:], mode='bilinear', align_corners=False), edgef) + y3)
        out2 = self.gelu(
            self.BGD2(F.interpolate(out3, size=y2.shape[2:], mode='bilinear', align_corners=False), edgef) + y2)
        out1 = self.gelu(
            self.BGD1(F.interpolate(out2, size=y1.shape[2:], mode='bilinear', align_corners=False), edgef) + y1)

        out1 = self.out_conv1(out1)
        out2 = self.out_conv2(out2)
        out3 = self.out_conv3(out3)
        out4 = self.out_conv4(out4)

        size = (out1.shape[2] * 8, out1.shape[3] * 8)
        out1 = F.interpolate(out1, size=size, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=size, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=size, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=size, mode='bilinear', align_corners=False)
        
        edge_feature = self.linearimg(edgef)

        return out1, out2, out3, out4, edge_feature
    
if __name__ == '__main__':
    from utils.tools import get_model_complexity

    model = FCHNet(Lchannels=(12,24,48,64), Hchannels=(36,72,144,192))

    flops, params = get_model_complexity(model, inputs=(torch.randn(size=(1, 3, 384, 384)),
                                                        torch.randn(size=(1, 96, 48, 48)),
                                                        torch.randn(size=(1, 96, 48, 48))),
                                         round=3)
    print(params, flops)