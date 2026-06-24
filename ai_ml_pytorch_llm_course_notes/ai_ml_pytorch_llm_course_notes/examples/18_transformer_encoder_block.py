"""Pre-Norm Transformer Encoder Block。"""
import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        h = self.norm1(x)
        attn_out, weights = self.attn(
            h, h, h,
            key_padding_mask=padding_mask,
            need_weights=True,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, weights


x = torch.randn(8, 20, 64)
block = EncoderBlock()
y, weights = block(x)
print(y.shape, weights.shape)
