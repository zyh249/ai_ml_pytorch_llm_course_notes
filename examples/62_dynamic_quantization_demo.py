"""PyTorch 动态量化：只对 Linear 层做 int8 量化。"""
from __future__ import annotations

import io
import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def serialized_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024**2


model = TinyClassifier().eval().cpu()
quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
x = torch.randn(4, 128)
print("fp32 output shape:", tuple(model(x).shape))
print("int8 output shape:", tuple(quantized(x).shape))
print("fp32 state_dict MB:", round(serialized_size_mb(model), 4))
print("int8 state_dict MB:", round(serialized_size_mb(quantized), 4))
