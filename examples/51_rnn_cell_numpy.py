"""NumPy 从零实现一个最小 RNN 单元。"""
from __future__ import annotations

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# 4 个时间步，每个输入向量 3 维；隐藏状态 2 维。
X = np.array([
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.2],
    [0.8, 0.4, 0.0],
    [0.1, 0.5, 1.0],
])
W_xh = np.array([[0.5, -0.3, 0.2], [0.1, 0.4, -0.2]])
W_hh = np.array([[0.6, 0.1], [-0.2, 0.5]])
b_h = np.zeros(2)
h = np.zeros(2)

for t, x_t in enumerate(X, start=1):
    h = np.tanh(W_xh @ x_t + W_hh @ h + b_h)
    print(f"t={t} x={x_t} h={h}")
