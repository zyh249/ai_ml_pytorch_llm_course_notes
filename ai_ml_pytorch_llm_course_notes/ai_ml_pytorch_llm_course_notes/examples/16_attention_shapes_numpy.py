"""Scaled Dot-Product Attention：完整展示每一步的形状。"""
import math
import numpy as np

np.set_printoptions(precision=3, suppress=True)
X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=float)  # [L=3, d_model=4]

# 为了便于理解，示例中直接令 Q=K=V=X。
Q = K = V = X
scores = Q @ K.T                         # [L, L]
scaled_scores = scores / math.sqrt(Q.shape[-1])
exp_scores = np.exp(scaled_scores - scaled_scores.max(axis=-1, keepdims=True))
weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
output = weights @ V                     # [L, d_v]

for name, value in {
    'X': X, 'Q': Q, 'K': K, 'V': V,
    'scores': scores, 'scaled_scores': scaled_scores,
    'weights': weights, 'output': output,
}.items():
    print(f'\n{name} shape={value.shape}\n{value}')
