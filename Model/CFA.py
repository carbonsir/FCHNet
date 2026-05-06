import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torchvision.transforms as transforms
from utils.dctNew import idct_tensor, dct_2d
from Model.Modules import DepthwiseSeparableConv, LightweightChannelAttention
from Model.RefineFM import RefineFM
from utils.hsvharr import feature_dwt, feature_idwt
from Model.HLTBlock import DWT,IDWT
class CFA(nn.Module):
    def __init__(self, Lin_channel, Hin_channel, Fintervals):
        super(CFA, self).__init__()
        with open('./utils/freq_mean_std.pkl', 'rb') as f:
            freq_stats = pickle.load(f)
        self.freq_norm = transforms.Normalize(mean=freq_stats['mean'], std=freq_stats['std'])

        self.cbam_begin = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Fintervals[0] , out_channels=Fintervals[0], 
                                          kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(Fintervals[0]),
            nn.GELU(), 
            DepthwiseSeparableConv(in_channels=Fintervals[0] , out_channels=Fintervals[1], 
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(Fintervals[1]),
            nn.GELU(), 
        )
        self.cbam_end = nn.Sequential(            
            DepthwiseSeparableConv(in_channels=Lin_channel * 2 , out_channels=Fintervals[1], 
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(Fintervals[1]),
            nn.GELU(), 
        )

        self.upC = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Lin_channel , out_channels=Lin_channel, 
                                          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Lin_channel),
            nn.GELU(),      
        )

        self.RFML3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Lin_channel , out_channels=Lin_channel, 
                                          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Lin_channel),
            nn.GELU(),      
        )
        self.RFML1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Lin_channel , out_channels=Lin_channel, 
                                          kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(Lin_channel),
            nn.GELU(),      
        )

        self.RFMH3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Hin_channel , out_channels=Hin_channel, 
                                          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Hin_channel),
            nn.GELU(), 
        )
        self.RFMH1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Hin_channel , out_channels=Hin_channel, 
                                          kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(Hin_channel),
            nn.GELU(), 
        )
        self.convtensor = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Lin_channel , out_channels=Lin_channel, 
                                          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Lin_channel),
            nn.GELU(),      
        )

        self.catconv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Lin_channel , out_channels=Lin_channel, 
                                          kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(Lin_channel),
            nn.GELU(),  
            RefineFM(Lin_channel)                   
        )
        self.catconv3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=Lin_channel , out_channels=Lin_channel, 
                                          kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(Lin_channel),
            nn.GELU(),  
            RefineFM(Lin_channel)                   
        )        
        self.dwt = DWT()
        self.idwt = IDWT()

    def forward(self, w_high, w_low, f_high, f_low):
        size = (w_low.shape[2] *2,  w_low.shape[2] *2)#6,96,96
        cba_freqH_up = F.interpolate(f_high, size=size, mode='bilinear', align_corners=False)
        cba_freqL_up = F.interpolate(f_low, size=size, mode='bilinear', align_corners=False)
    
        cba_freqH = self.cbam_begin(cba_freqH_up)#6,48,48
        cba_freqL = self.cbam_begin(cba_freqL_up)
        ifftF = idct_tensor(cba_freqH, cba_freqL) #12,96,96
        upifftF = self.upC(ifftF)
        iffToWL, iffToWH = self.dwt(upifftF)

        if w_low.shape[2] < 12:
            L_att = self.RFML1(iffToWL + w_low)     
            H_att = self.RFMH1(iffToWH + w_high)
        else:
            L_att = self.RFML3(iffToWL + w_low)     
            H_att = self.RFMH3(iffToWH + w_high)

        idwt_sum = self.idwt(L_att, H_att)

        rgbtensor = self.convtensor(idwt_sum)

        freq_sum = dct_2d(rgbtensor)
        high_sum, low_sum = self.freq_decompose(freq_sum)

        cba_freqH_down = F.interpolate(cba_freqH, size=high_sum.shape[2:], mode='bilinear', align_corners=False)
        cba_freqL_down = F.interpolate(cba_freqL, size=high_sum.shape[2:], mode='bilinear', align_corners=False)

        resH = self.cbam_end(high_sum) + cba_freqH_down
        resL = self.cbam_end(low_sum) + cba_freqL_down

        cat_HL = torch.cat([resH, resL], dim=1)

        if cat_HL.shape[2] < 12:
            y = self.catconv1(cat_HL)
        else:
            y = self.catconv3(cat_HL)

        if w_low.shape[1] == 12:
            edge = idct_tensor(resH, resL)
            return y, edge
        return y, resH
    def freq_decompose(self, freq):
        PE = freq.shape[1] 
        P = PE // 3
        PY = P // 2
        freq_y = freq[:, 0:P, :, :]
        freq_Cb = freq[:,P:P*2, :, :]
        freq_Cr = freq[:,P*2:PE, :, :]
        high = torch.cat([freq_y[:, PY:, :, :], freq_Cb[:, PY:, :, :], freq_Cr[:, PY:, :, :]], dim=1)
        low = torch.cat([freq_y[:, :PY, :, :], freq_Cb[:, :PY, :, :], freq_Cr[:, :PY, :, :]], dim=1)
        return high, low    
    