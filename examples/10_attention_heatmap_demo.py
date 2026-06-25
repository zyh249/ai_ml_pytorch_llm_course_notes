"""自注意力简化实现：输出注意力权重矩阵。"""
import math
import numpy as np

X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.5, 0.5, 1.0, 0.0],
], dtype=float)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


Q = K = V = X
scores = Q @ K.T / math.sqrt(Q.shape[-1])
weights = softmax(scores)
out = weights @ V

np.set_printoptions(precision=3, suppress=True)
print("scores =")
print(scores)
print("\nweights =")
print(weights)
print("\noutput =")
print(out)
