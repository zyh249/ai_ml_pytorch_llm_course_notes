"""NumPy 从零实现二维卷积（更准确地说是互相关）。"""
import numpy as np

image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
], dtype=float)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=float)

h_out = image.shape[0] - kernel.shape[0] + 1
w_out = image.shape[1] - kernel.shape[1] + 1
output = np.zeros((h_out, w_out))
for i in range(h_out):
    for j in range(w_out):
        patch = image[i:i+3, j:j+3]
        output[i, j] = np.sum(patch * kernel)
print(output)
