"""KNN 逐步演示：输出每个样本与查询点的距离、排序结果和最终投票。"""
from collections import Counter
from math import sqrt

points = [
    ("A", (1.0, 2.0), "蓝色类"),
    ("B", (2.0, 1.5), "蓝色类"),
    ("C", (3.0, 3.5), "红色类"),
    ("D", (3.6, 2.8), "红色类"),
    ("E", (1.8, 3.2), "蓝色类"),
    ("F", (4.2, 1.2), "红色类"),
]
query = (2.8, 2.4)
k = 3


def euclidean(p1, p2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


records = []
for name, coord, label in points:
    d = euclidean(coord, query)
    records.append((d, name, coord, label))

records.sort(key=lambda x: x[0])
print("查询点:", query)
print("\n距离排序结果:")
for d, name, coord, label in records:
    print(f"{name} {coord} -> 距离={d:.3f}, 类别={label}")

neighbors = records[:k]
votes = Counter(label for _, _, _, label in neighbors)
pred = votes.most_common(1)[0][0]
print(f"\n前 {k} 个近邻:")
for d, name, coord, label in neighbors:
    print(f"  {name} {coord} {label} 距离={d:.3f}")
print("预测类别:", pred)
