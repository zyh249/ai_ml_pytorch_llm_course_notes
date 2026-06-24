"""MSE、BCEWithLogitsLoss 与 CrossEntropyLoss。"""
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.set_num_threads(1)
print("MSE =", nn.MSELoss()(torch.tensor([2.5,4.0]), torch.tensor([3.0,5.0])).item())
print("BCEWithLogits =", nn.BCEWithLogitsLoss()(torch.tensor([1.4,-0.8]), torch.tensor([1.,0.])).item())
logits = torch.tensor([[1.2,2.4,0.2],[2.0,0.5,1.1]])
target = torch.tensor([1,0], dtype=torch.long)
print("CrossEntropy =", nn.CrossEntropyLoss()(logits,target).item())
