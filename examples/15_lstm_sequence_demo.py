"""LSTM 序列建模示例：观察输入输出形状。"""
import torch
from torch import nn

x = torch.randn(5, 12, 16)  # batch=5, seq_len=12, feature_dim=16
lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
out, (h_n, c_n) = lstm(x)
print('input shape:', x.shape)
print('output shape:', out.shape)
print('hidden shape:', h_n.shape)
print('cell shape:', c_n.shape)
