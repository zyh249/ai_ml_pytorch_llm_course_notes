"""LSTM 与 GRU 的状态、形状和参数量。"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

x = torch.randn(3, 5, 10)  # [batch, seq_len, input_size]

lstm = nn.LSTM(10, 16, num_layers=2, batch_first=True, bidirectional=True)
lstm_out, (h_n, c_n) = lstm(x)
print("LSTM output:", lstm_out.shape)
print("LSTM h_n:", h_n.shape)
print("LSTM c_n:", c_n.shape)
print("LSTM params:", sum(p.numel() for p in lstm.parameters()))

gru = nn.GRU(10, 16, num_layers=2, batch_first=True, bidirectional=True)
gru_out, gru_h = gru(x)
print("GRU output:", gru_out.shape)
print("GRU h_n:", gru_h.shape)
print("GRU params:", sum(p.numel() for p in gru.parameters()))
