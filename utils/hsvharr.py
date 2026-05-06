import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import torch
from pytorch_wavelets import DWTForward, DWTInverse

def feature_dwt(image_tensor, wavelet='haar', level=1):
    assert image_tensor.dim() == 4 , "4-dim"
    
    dwt = DWTForward(wave=wavelet, J=level, mode='zero').to(image_tensor.device)
    
    yl, yh = dwt(image_tensor)  # yl:[32,3,6,6], yh:[(cH,cV,cD)]
    
    low_freq = yl

    last_level = yh[-1]
    
    H = image_tensor.shape[2] // 2
    C = image_tensor.shape[1] 
    B = image_tensor.shape[0]

    high_freq = last_level.reshape(B, C * 3, H, H)

    return low_freq, high_freq

def feature_idwt(low_freq, high_freq, wavelet='haar', level=1):

    device = low_freq.device
    B, C, H, W = low_freq.shape
    

    assert high_freq.shape == (B, C*3, H, W), \
        f"高频分量形状应为 {(B, C*3, H, W)}，但得到 {high_freq.shape}"
    

    high_reshaped = high_freq.reshape(B, C, 3, H, W)  # [B,C,3,H,W]
    cH = high_reshaped[:, :, 0]  # [B,C,H,W]
    cV = high_reshaped[:, :, 1]
    cD = high_reshaped[:, :, 2]
    

    idwt = DWTInverse(wave=wavelet, mode='zero').to(device)

    yh_channels = []
    for c in range(C):
        channel_components = (
            cH[:, c].unsqueeze(1),  # (B,1,H,W)
            cV[:, c].unsqueeze(1),
            cD[:, c].unsqueeze(1)
        )
        yh_channels.append([channel_components])  
    

    reconstructed = []
    for c in range(C):
        yl = low_freq[:, c].unsqueeze(1)  # (B,1,H,W)
        

        cH, cV, cD = yh_channels[c][0]  
        yh_tensor = torch.stack([cH, cV, cD], dim=2)  # (B,1,3,H,W)
        
        rec = idwt((yl, [yh_tensor]))  
        reconstructed.append(rec.squeeze(1))
    
    return torch.stack(reconstructed, dim=1)
