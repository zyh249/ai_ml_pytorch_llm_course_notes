"""对比不同优化器在同一个简单分类任务上的使用方式。"""
import torch
from torch import nn

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x):
        return self.net(x)

x = torch.randn(64, 10)
y = torch.randint(0, 3, (64,))
loss_fn = nn.CrossEntropyLoss()

def train_one_optimizer(name, optimizer_factory):
    torch.manual_seed(0)
    model = TinyClassifier()
    optimizer = optimizer_factory(model.parameters())
    for step in range(20):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'{name:10s} final loss = {loss.item():.4f}')

train_one_optimizer('SGD', lambda p: torch.optim.SGD(p, lr=0.05))
train_one_optimizer('Momentum', lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9))
train_one_optimizer('AdaGrad', lambda p: torch.optim.Adagrad(p, lr=0.05))
train_one_optimizer('RMSProp', lambda p: torch.optim.RMSprop(p, lr=0.005, alpha=0.99))
train_one_optimizer('Adam', lambda p: torch.optim.Adam(p, lr=0.005))
train_one_optimizer('AdamW', lambda p: torch.optim.AdamW(p, lr=0.005, weight_decay=0.01))
