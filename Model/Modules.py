import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, bias=True, se_ratio=2):
        super(DepthwiseSeparableConv, self).__init__()

        if dilation != 1:
            padding = dilation

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.se = SqueezeExcitation(in_channels=in_channels, out_channels=in_channels, reduction=se_ratio)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = torch.relu(self.bn(out))
        out = self.se(out)
        out = self.pointwise(out)
        return out


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super(ConvBN, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNGeLU(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super(ConvBNGeLU, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class SqueezeExcitation(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduction: int = 4,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid,
            pool='avgpool'
    ):
        super(SqueezeExcitation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.transition = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        if out_channels // reduction == 0:
            reduction = 1

        if pool == 'avgpool':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'maxpool':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            print('Parameter pool is not avgpool or maxpool')
            return
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.transition(x)
        scale = self._scale(x)
        return scale * x
class LightweightChannelAttention(nn.Module):
    def __init__(self, channel):
        super(LightweightChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // 3, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        y = self.sigmoid(avg_out + max_out)
        return x * y 