"""CIFAR-10 CNN 教学骨架。默认只打印网络；传入 --train 才下载并训练。"""
import sys
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit('请先安装 PyTorch：pip install torch torchvision') from exc

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),   # 32 -> 30
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 30 -> 15
            nn.Conv2d(6, 16, kernel_size=3),  # 15 -> 13
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 13 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SmallCNN()
print(model)
print('output shape:', model(torch.randn(4, 3, 32, 32)).shape)

if '--train' in sys.argv:
    print('请在此处接入 torchvision.datasets.CIFAR10、DataLoader、交叉熵和优化器。')
