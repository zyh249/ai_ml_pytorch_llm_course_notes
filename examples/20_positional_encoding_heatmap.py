"""生成正弦余弦位置编码，并打印一个小型热力矩阵。"""
import math
import numpy as np


def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model), dtype=float)
    position = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div)
    pe[:, 1::2] = np.cos(position * div)
    return pe

pe = positional_encoding(max_len=8, d_model=12)
np.set_printoptions(precision=3, suppress=True)
print('shape =', pe.shape)
print(pe)
