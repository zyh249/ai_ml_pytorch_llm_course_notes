"""Xavier 与 Kaiming 初始化及前向方差观察。"""
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.set_num_threads(1)
def build(depth=8, width=128, method="kaiming"):
    layers = []
    for _ in range(depth):
        linear = nn.Linear(width, width)
        if method == "xavier": nn.init.xavier_normal_(linear.weight)
        elif method == "kaiming": nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        elif method == "zero": nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ReLU()]
    return nn.Sequential(*layers)

x = torch.randn(512, 128)
for method in ["zero", "xavier", "kaiming"]:
    y, variances = x, []
    for layer in build(method=method):
        y = layer(y)
        if isinstance(layer, nn.ReLU): variances.append(y.var().item())
    print(method, [round(v,4) for v in variances])
