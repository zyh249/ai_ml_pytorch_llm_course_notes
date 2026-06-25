"""PyTorch RNN 输入输出形状、堆叠与双向设置。"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

batch, seq_len, input_size = 4, 6, 8
hidden_size, num_layers = 12, 2

rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,
)
x = torch.randn(batch, seq_len, input_size)
h0 = torch.zeros(num_layers * 2, batch, hidden_size)
output, h_n = rnn(x, h0)

print("x:", x.shape)
print("output:", output.shape)  # [batch, seq_len, hidden_size * directions]
print("h_n:", h_n.shape)       # [layers * directions, batch, hidden_size]
