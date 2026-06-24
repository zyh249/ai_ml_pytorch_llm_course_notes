"""Transformer 正弦余弦位置编码。"""
import math
try:
    import torch
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = sinusoidal_positional_encoding(max_len=6, d_model=8)
print(pe.round(decimals=4))
