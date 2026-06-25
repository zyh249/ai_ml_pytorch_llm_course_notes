"""NumPy 线性回归：用梯度下降拟合 y = 3x + 2。"""
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 80)
y = 3 * x + 2 + rng.normal(0, 1.2, size=x.shape)

w, b = 0.0, 0.0
lr = 0.01
for epoch in range(1000):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    grad_w = 2 * np.mean((y_hat - y) * x)
    grad_b = 2 * np.mean(y_hat - y)
    w -= lr * grad_w
    b -= lr * grad_b
    if epoch in (0, 1, 2, 9, 99, 999):
        print(f"epoch={epoch:>3} loss={loss:>7.3f} w={w:>6.3f} b={b:>6.3f}")
