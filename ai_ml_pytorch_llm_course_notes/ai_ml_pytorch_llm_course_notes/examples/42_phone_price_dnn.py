"""合成手机参数数据上的四分类 DNN 案例。"""
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.set_num_threads(1); torch.manual_seed(42)
X=torch.randn(2000,20); score=1.6*X[:,0]+1.1*X[:,1]-0.8*X[:,2]+0.4*X[:,3]
y=torch.bucketize(score,torch.quantile(score,torch.tensor([.25,.5,.75])))
perm=torch.randperm(len(X)); tr,va=perm[:1600],perm[1600:]
loader=DataLoader(TensorDataset(X[tr],y[tr]),batch_size=64,shuffle=True)
valid=TensorDataset(X[va],y[va])
model=nn.Sequential(nn.Linear(20,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Linear(256,4))
loss_fn=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
for _ in range(8):
    model.train()
    for xb,yb in loader:
        loss=loss_fn(model(xb),yb); opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    xv,yv=valid.tensors; acc=(model(xv).argmax(1)==yv).float().mean().item()
print("validation accuracy =",round(acc,4))
