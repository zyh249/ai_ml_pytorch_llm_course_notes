"""PyTorch 动态量化示例：Linear 层从 FP32 转为 INT8。"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请安装：pip install torch") from exc

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).eval()
quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
x = torch.randn(4, 128)
with torch.inference_mode():
    print("FP32 output:", model(x).shape)
    print("INT8 output:", quantized(x).shape)
print(quantized)
