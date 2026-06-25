"""Dropout 与 BatchNorm 的训练/推理差异。"""
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.set_num_threads(1); torch.manual_seed(0)
x=torch.tensor([[1.,2.,3.,4.],[2.,4.,6.,8.]])
drop=nn.Dropout(0.5); print("dropout train:\n",drop(x)); drop.eval(); print("dropout eval:\n",drop(x))
bn=nn.BatchNorm1d(4); bn.train(); print("batchnorm train:\n",bn(x)); bn.eval(); print("batchnorm eval:\n",bn(x))
