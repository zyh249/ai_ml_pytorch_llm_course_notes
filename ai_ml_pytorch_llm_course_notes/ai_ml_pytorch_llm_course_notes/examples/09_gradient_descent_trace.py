"""一维梯度下降轨迹：观察参数如何沿损失曲线下降。"""

def f(w):
    return (w - 2) ** 2 + 1

def grad(w):
    return 2 * (w - 2)

w = -4.0
lr = 0.2
print(f'初始: w={w:.3f}, loss={f(w):.3f}')
for step in range(1, 13):
    g = grad(w)
    w = w - lr * g
    print(f'step={step:>2} grad={g:>7.3f} w={w:>7.3f} loss={f(w):>7.3f}')
