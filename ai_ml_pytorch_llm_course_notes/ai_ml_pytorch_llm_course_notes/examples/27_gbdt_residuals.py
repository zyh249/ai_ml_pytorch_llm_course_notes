"""GBDT 回归的核心直觉：每轮拟合上一轮的残差。"""
import numpy as np

# 为了突出思想，用常数弱学习器代替真正的回归树。
y = np.array([90.0, 70.0, 50.0, 30.0])
pred = np.full_like(y, y.mean())
learning_rate = 0.5
print('initial prediction:', pred)
for round_id in range(1, 5):
    residual = y - pred
    weak_prediction = np.full_like(y, residual.mean())
    pred = pred + learning_rate * weak_prediction
    print(f'round={round_id}')
    print(' residual:', residual)
    print(' prediction:', pred)
