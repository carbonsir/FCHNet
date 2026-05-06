import torch
import torch.nn.functional as F
import torch_dct as dct


def rgb2ycbcr(rgb_tensor):
    if len(rgb_tensor.shape) != 4 or rgb_tensor.shape[1] != 3:
        raise ValueError("input image is not a rgb tensor: %s" % str(rgb_tensor.shape))
    rgb_tensor = rgb_tensor.to(torch.float32)

    transform_matrix = torch.tensor([[0.257, 0.564, 0.098],
                                     [-0.148, -0.291, 0.439],
                                     [0.439, -0.368, -0.071]]).to(rgb_tensor.device)

    shift_matrix = torch.tensor([16, 128, 128]).reshape(-1, 1).to(rgb_tensor.device)

    ycbcr_tensor = torch.matmul(transform_matrix, rgb_tensor.flatten(2)) + shift_matrix
    ycbcr_tensor = ycbcr_tensor.reshape(rgb_tensor.shape)
    return ycbcr_tensor


def rgb2gray(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = torch.unsqueeze(gray, 1)
    return gray


def dct_2d(rgb_tensor, P=8):
    """
    Modified according to https://github.com/VisibleShadow/Implementation-of-Detecting-Camouflaged-Object-in-Frequency-Domain/blob/main/train.py
    """
    ycbcr_tensor = rgb2ycbcr(rgb_tensor)
    num_batchsize = ycbcr_tensor.shape[0]
    size = ycbcr_tensor.shape[2]
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, 3, size // P, P, size // P, P).permute(0, 2, 4, 1, 3, 5)
    ycbcr_tensor = dct.dct_2d(ycbcr_tensor, norm='ortho')
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, size // P, size // P, -1).permute(0, 3, 1, 2)
    return ycbcr_tensor

def idct_2d(freq_tensor, P=8):

    num_batchsize = freq_tensor.shape[0]
    channels = freq_tensor.shape[1]
    size = freq_tensor.shape[2] * P  # 原始图像尺寸
    
    freq_tensor = freq_tensor.reshape(num_batchsize, size//P, size//P, 3, P, P)
    freq_tensor = freq_tensor.permute(0, 3, 1, 4, 2, 5)  # 重排维度
    
    ycbcr_tensor = dct.idct_2d(freq_tensor, norm='ortho')
    
    ycbcr_tensor = ycbcr_tensor.permute(0, 1, 2, 4, 3, 5)
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, 3, size, size)
    
    return ycbcr_tensor

def freq_recompose(high, low):

    high_y, high_Cb, high_Cr = torch.split(high, [32, 32, 32], dim=1)
    low_y, low_Cb, low_Cr = torch.split(low, [32, 32, 32], dim=1)
    
    freq_y = torch.cat([low_y, high_y], dim=1)
    freq_Cb = torch.cat([low_Cb, high_Cb], dim=1)
    freq_Cr = torch.cat([low_Cr, high_Cr], dim=1)
    
    freq = torch.cat([freq_y, freq_Cb, freq_Cr], dim=1)
    return freq

def ycbcr2rgb(ycbcr_tensor):

    matrix = torch.tensor([[1, 0, 1.402],
                          [1, -0.344136, -0.714136],
                          [1, 1.772, 0]], 
                         dtype=ycbcr_tensor.dtype, device=ycbcr_tensor.device)
    
    ycbcr_tensor = ycbcr_tensor.clone()
    ycbcr_tensor[:, 1, :, :] -= 128/255
    ycbcr_tensor[:, 2, :, :] -= 128/255
    
    rgb_tensor = torch.einsum('cv,bvxy->bcxy', matrix, ycbcr_tensor)
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
    return rgb_tensor

def idct_tensor(high, low, P=8):
    freq = freq_recompose(high, low)

    freq = freq * 7.0

    ycbcr_tensor = idct_2d(freq, P)

    return ycbcr_tensor
def reconstruct_img(high, low, P=8):
    freq = freq_recompose(high, low)
    freq = freq * 7.0
    freq = freq.unsqueeze(0)
    ycbcr_tensor = idct_2d(freq, P)
    rgb_tensor = ycbcr2rgb(ycbcr_tensor)
    
    return rgb_tensor.squeeze(0)  