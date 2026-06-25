"""PyTorch 常见学习率衰减策略。"""
try:
    import torch
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.set_num_threads(1)
p=torch.nn.Parameter(torch.tensor(1.0))
for name in ["step","multi","exp"]:
    opt=torch.optim.SGD([p],lr=0.1)
    sch=(torch.optim.lr_scheduler.StepLR(opt,4,0.5) if name=="step" else
         torch.optim.lr_scheduler.MultiStepLR(opt,[3,7],0.5) if name=="multi" else
         torch.optim.lr_scheduler.ExponentialLR(opt,0.9))
    vals=[]
    for _ in range(10): vals.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    print(name,[round(v,5) for v in vals])
