"""全局非结构化剪枝：演示稀疏度，而非保证真实加速。"""
from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.utils.prune as prune
except ImportError as exc:
    raise SystemExit("请安装：pip install torch") from exc

model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
parameters = [(model[0], "weight"), (model[2], "weight")]
prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=0.30)
for module, name in parameters:
    prune.remove(module, name)
weights = torch.cat([model[0].weight.flatten(), model[2].weight.flatten()])
print("sparsity =", float((weights == 0).float().mean()))
print("注意：标准稠密算子通常不会因为出现 0 就自动提速。")
