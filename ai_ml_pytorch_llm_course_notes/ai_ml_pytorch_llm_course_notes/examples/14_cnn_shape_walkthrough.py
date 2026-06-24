"""CNN 形状变化示例。"""
import torch
from torch import nn

x = torch.randn(8, 3, 32, 32)  # batch=8, RGB 图像
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

y = x
print('input:', y.shape)
for layer in model:
    y = layer(y)
    print(layer.__class__.__name__, '->', y.shape)
