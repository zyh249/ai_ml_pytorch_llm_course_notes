"""激活函数、初始化、损失函数、Dropout/BatchNorm 小点汇总示例。"""
import torch
from torch import nn
import torch.nn.functional as F

x = torch.linspace(-3, 3, 7)
print('ReLU:', F.relu(x))
print('GELU:', F.gelu(x))

model = nn.Sequential(
    nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.3),
    nn.BatchNorm1d(16), nn.Linear(16, 3)
)
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

inputs = torch.randn(5, 8)
labels = torch.tensor([0, 1, 2, 1, 0])
logits = model(inputs)
loss = nn.CrossEntropyLoss()(logits, labels)
print('logits shape:', logits.shape)
print('loss:', loss.item())

model.eval()
with torch.no_grad():
    print('eval output shape:', model(inputs).shape)
