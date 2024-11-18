import numpy as np
import torch
import torch.nn as nn
import math

# class SimpleSelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleSelfAttention, self).__init__()
#
#         self.query = nn.Linear(in_channels, in_channels)
#         self.key = nn.Linear(in_channels, in_channels)
#         self.value = nn.Linear(in_channels, in_channels)
#         self.softmax = nn.Softmax(dim=-1)
#         self.norm_layer = nn.LayerNorm(in_channels)
#
#     def forward(self, x):
#         B, H, W, C = x.size()
#         x = x.view(B, H * W, C)  # Flatten to sequence
#         q, k, v = self.query(x), self.key(x), self.value(x)
#
#         attention_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5))
#         # out = torch.bmm(attention_weights, v).view(B, H, W, C)
#         out = torch.bmm(attention_weights, v)  # (B, H*W, C)
#
#         out = self.norm_layer(out)
#
#         return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        d_k = q.size(-1)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.dropout((attn))
        output = torch.matmul(self.softmax(attn), v)
        attn = attn[:, :, -1]
        return output, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = in_channels // num_heads

        # Query, Key, Value 생성
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.norm_layer = nn.LayerNorm(in_channels, eps=1e-6)

    def forward(self, x):
        B, H, W, C = x.size()
        x = x.view(B, H * W, C)  # Flatten (B, H*W, C)

        # Query, Key, Value 계산
        q = self.query(x).view(B, H * W, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.key(x).view(B, H * W, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.value(x).view(B, H * W, self.num_heads, self.dim_per_head).transpose(1, 2)

        # residual 추가
        residual = v.view(B, H * W, -1)

        out, attn = self.attention(q, k, v)

        # Head 결합
        attn_output = out.transpose(1, 2).contiguous().view(B, H * W, C)

        output = self.dropout(self.out_proj(attn_output))  # (B, H * W, C)

        output += residual

        output = self.norm_layer(output)

        return output