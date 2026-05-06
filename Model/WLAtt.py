import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Modules import LightweightChannelAttention
from Model.CBAM import SpatialAttention

class WLAtt(nn.Module):
    def __init__(self, channel):
        super(WLAtt, self).__init__()
        self.channel_attn = LightweightChannelAttention(channel)
        self.non_local = NonLocalAttention(channel)
        self.spatial_attn = SpatialAttention()
    def forward(self, xl):
        ll = self.channel_attn(xl)
        ll = self.non_local(ll)
        ll = xl * self.spatial_attn(ll)

        return ll
class NonLocalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels//2, 1)
        self.phi = nn.Conv2d(in_channels, in_channels//2, 1)
        self.g = nn.Conv2d(in_channels, in_channels//2, 1)
        self.out_conv = nn.Conv2d(in_channels//2, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        theta = self.theta(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        phi = self.phi(x).view(batch_size, -1, height*width)
        g = self.g(x).view(batch_size, -1, height*width)
        
        attention = torch.bmm(theta, phi)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        out = self.out_conv(out)
        
        return self.gamma * out + x    