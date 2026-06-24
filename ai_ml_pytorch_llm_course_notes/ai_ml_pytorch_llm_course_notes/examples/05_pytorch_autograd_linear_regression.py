"""PyTorch 自动微分 + nn.Module 线性回归完整训练流程。"""
try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.manual_seed(0)
X = torch.linspace(-3, 3, 120).unsqueeze(1)
y = 2.0 * X - 0.5 + torch.randn_like(X) * 0.25

loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(60):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()  # 清空上一轮梯度
        loss.backward()        # 反向传播，计算参数梯度
        optimizer.step()       # 根据梯度更新参数
    if epoch in (0, 1, 2, 9, 59):
        w = model.weight.item()
        b = model.bias.item()
        print(f"epoch={epoch:>2} loss={loss.item():.4f} w={w:.3f} b={b:.3f}")
