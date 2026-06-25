"""非结构化剪枝：制造稀疏权重，并解释为什么不一定带来真实加速。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import prune

layer = nn.Linear(32, 16)
prune.l1_unstructured(layer, name="weight", amount=0.30)
prune.remove(layer, "weight")
zero_count = int((layer.weight == 0).sum().item())
total = layer.weight.numel()
print(f"sparsity = {zero_count / total:.2%}")
print("output shape =", tuple(layer(torch.randn(4, 32)).shape))
print("注意：普通稠密 Linear 内核仍可能计算这些零；要加速需稀疏内核或结构化剪枝。")
