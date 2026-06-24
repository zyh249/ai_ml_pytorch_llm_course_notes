"""最大池化与平均池化。"""
import numpy as np

x = np.array([
    [1, 3, 2, 0],
    [4, 6, 5, 1],
    [2, 1, 7, 3],
    [0, 2, 4, 8],
], dtype=float)


def pool(x, mode='max', kernel=2, stride=2):
    h_out = (x.shape[0] - kernel) // stride + 1
    w_out = (x.shape[1] - kernel) // stride + 1
    out = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = x[i*stride:i*stride+kernel, j*stride:j*stride+kernel]
            out[i, j] = patch.max() if mode == 'max' else patch.mean()
    return out

print('max pool:\n', pool(x, 'max'))
print('avg pool:\n', pool(x, 'avg'))
