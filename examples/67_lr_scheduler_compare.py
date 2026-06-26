"""学习率调度器示例：观察每轮 lr 如何变化。"""
import torch
from torch import nn

model = nn.Linear(4, 2)

def show_scheduler(name, scheduler_factory):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = scheduler_factory(optimizer)
    print('\n' + name)
    for epoch in range(8):
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            val_loss = 1.0 if epoch < 3 else 0.9  # 模拟验证集平台期
            scheduler.step(val_loss)
        else:
            scheduler.step()
        print(epoch, optimizer.param_groups[0]['lr'])

show_scheduler('StepLR', lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5))
show_scheduler('ExponentialLR', lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9))
show_scheduler('CosineAnnealingLR', lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8))
show_scheduler('ReduceLROnPlateau', lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2))
