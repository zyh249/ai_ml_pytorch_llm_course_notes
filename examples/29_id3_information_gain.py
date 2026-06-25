"""ID3：计算标签熵、条件熵与信息增益。"""
import math
from collections import Counter, defaultdict


def entropy(labels):
    n = len(labels)
    return -sum((c / n) * math.log2(c / n) for c in Counter(labels).values())


def information_gain(feature, labels):
    base = entropy(labels)
    groups = defaultdict(list)
    for x, y in zip(feature, labels):
        groups[x].append(y)
    conditional = sum(len(group) / len(labels) * entropy(group) for group in groups.values())
    return base - conditional

labels = ['买', '买', '不买', '不买', '买', '不买']
age = ['青年', '青年', '青年', '中年', '中年', '老年']
income = ['高', '低', '低', '高', '高', '低']
print('H(Y) =', round(entropy(labels), 4))
print('Gain(age) =', round(information_gain(age, labels), 4))
print('Gain(income) =', round(information_gain(income, labels), 4))
