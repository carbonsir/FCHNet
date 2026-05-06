import torch
import torch.nn as nn
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        hht = torch.cat([lh, hl, hh], dim=1)
        
        return ll, hht
    
class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, ll, hht):
        batch_size, channels_times_3, height, width = hht.shape
        channels = channels_times_3 // 3
        
        lh = hht[:, :channels, :, :]
        hl = hht[:, channels:2*channels, :, :]
        hh = hht[:, 2*channels:, :, :]
        
        x1 = (ll - lh - hl + hh) / 2
        x2 = (ll + lh - hl - hh) / 2
        x3 = (ll - lh + hl - hh) / 2
        x4 = (ll + lh + hl + hh) / 2
        
        batch_size, channels, height, width = ll.shape
        x = torch.zeros(batch_size, channels, height * 2, width * 2, 
                       dtype=ll.dtype, device=ll.device)
        
        x[:, :, 0::2, 0::2] = x1
        x[:, :, 1::2, 0::2] = x2
        x[:, :, 0::2, 1::2] = x3
        x[:, :, 1::2, 1::2] = x4
        
        return x
class BasicBlockL(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlockL, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups)
        self.conv11 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                  padding=0, groups=groups)
        
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=3)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 =  nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups)
        self.conv21 =  nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                  padding=0, groups=groups)        
        self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=3)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x
        if x.shape[2] >6:
            out1 = self.conv1(x)
        else:
            out1 = self.conv11(x)
        out1 = self.conv1_1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 += identity
        if x.shape[2] >6:        
            out2 = self.conv2(out1)
        else:
            out2 = self.conv21(out1)
        out2 = self.conv2_1(out2)
        out2 = self.bn2(out2)
        out = self.relu(out2)

        return out

class  BasicBlockH(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlockH, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1,3), padding=(0, 1))
        self.conv1_2 = nn.Conv2d(inplanes, planes, kernel_size=(3,1), padding=(1, 0))
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), padding=(0, 1))
        self.conv2_2 = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        out1 = self.conv1_2(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out = identity + out1

        out = self.conv2(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out