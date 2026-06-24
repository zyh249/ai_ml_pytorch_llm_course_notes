"""决策树划分指标：熵与基尼指数。"""
import math

def entropy(labels):
    total = len(labels)
    counts = {v: labels.count(v) for v in set(labels)}
    return -sum((c/total) * math.log2(c/total) for c in counts.values())


def gini(labels):
    total = len(labels)
    counts = {v: labels.count(v) for v in set(labels)}
    return 1 - sum((c/total) ** 2 for c in counts.values())

root = [1, 1, 1, 0, 0, 0, 0, 1]
left = [1, 1, 1, 1]
right = [0, 0, 0, 1]

print('root entropy =', round(entropy(root), 4))
print('left entropy =', round(entropy(left), 4))
print('right entropy =', round(entropy(right), 4))
print('root gini =', round(gini(root), 4))
print('left gini =', round(gini(left), 4))
print('right gini =', round(gini(right), 4))
