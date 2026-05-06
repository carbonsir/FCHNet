import torch
import torch.nn as nn
class SimplifiedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SimplifiedSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_size, heads)
        
    def forward(self, x, mask=None):
        x = x.transpose(0, 1)
        print(x.shape)
        attn_output, attn_weights = self.multihead_attn(x, x, x, attn_mask=mask)
        return attn_output.transpose(0, 1), attn_weights