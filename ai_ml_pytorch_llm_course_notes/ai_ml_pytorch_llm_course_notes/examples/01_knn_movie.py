"""KNN 电影类型预测：不依赖第三方库，便于理解距离、近邻和投票。"""
from collections import Counter
from math import sqrt

movies = [
    ("功夫熊猫", [39, 0, 31], "喜剧片"),
    ("叶问3", [3, 2, 65], "动作片"),
    ("伦敦陷落", [2, 3, 55], "动作片"),
    ("代理情人", [9, 38, 2], "爱情片"),
    ("新步步惊心", [8, 34, 17], "爱情片"),
    ("谍影重重", [5, 2, 57], "动作片"),
    ("美人鱼", [21, 17, 5], "喜剧片"),
    ("宝贝当家", [45, 2, 9], "喜剧片"),
]
unknown = ("唐人街探案", [23, 3, 17])

def euclidean(a, b):
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn_predict(dataset, sample, k=3):
    distances = []
    for name, features, label in dataset:
        distances.append((euclidean(features, sample), name, label))
    distances.sort(key=lambda item: item[0])
    neighbors = distances[:k]
    votes = Counter(label for _, _, label in neighbors)
    return votes.most_common(1)[0][0], neighbors

if __name__ == "__main__":
    pred, nearest = knn_predict(movies, unknown[1], k=3)
    print(f"预测电影：{unknown[0]}")
    print("最近邻：")
    for dist, name, label in nearest:
        print(f"  {name:<8} 距离={dist:.2f} 类型={label}")
    print("预测类型：", pred)
