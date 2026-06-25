"""文本语料分析：标签分布、句长分布、覆盖率和高频词。"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median

samples = [
    ("房间干净，服务很好", 1),
    ("早餐太少，隔音很差", 0),
    ("位置方便，下次还来", 1),
    ("前台态度一般", 0),
    ("景色很好，交通便利", 1),
    ("空调噪声比较大", 0),
]

labels = Counter(label for _, label in samples)
lengths = [len(text) for text, _ in samples]
chars = Counter(char for text, _ in samples for char in text if char.strip() and char not in "，。！？")

print("标签分布:", labels)
print("句长: min/max/mean/median =", min(lengths), max(lengths), round(mean(lengths), 2), median(lengths))
for max_len in [6, 8, 10, 12]:
    coverage = sum(length <= max_len for length in lengths) / len(lengths)
    print(f"max_len={max_len:>2} 覆盖率={coverage:.1%}")
print("高频字符:", chars.most_common(10))
