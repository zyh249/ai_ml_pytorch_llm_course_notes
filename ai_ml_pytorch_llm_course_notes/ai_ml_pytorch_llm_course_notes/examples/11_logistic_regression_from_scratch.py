"""逻辑回归核心计算：sigmoid + 二分类概率。"""
import math

samples = [
    ([1.0, 2.0], 0),
    ([1.5, 1.8], 0),
    ([3.0, 3.2], 1),
    ([3.5, 4.0], 1),
]
# 假设已经训练得到一组参数
w = [1.2, 1.0]
b = -4.0

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

for x, y in samples:
    z = sum(a * b for a, b in zip(x, w)) + b
    p = sigmoid(z)
    pred = 1 if p >= 0.5 else 0
    print(f'x={x}, z={z:.3f}, p={p:.3f}, pred={pred}, y={y}')
