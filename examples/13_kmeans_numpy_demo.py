"""KMeans 聚类简化版：初始化中心 -> 分配样本 -> 更新中心。"""
import numpy as np

X = np.array([
    [1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
    [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]
])
centers = np.array([[1.0, 1.0], [5.0, 7.0]])

for step in range(1, 5):
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    new_centers = np.array([X[labels == k].mean(axis=0) for k in range(len(centers))])
    print(f"step={step}")
    print("labels =", labels.tolist())
    print("centers =\n", new_centers)
    if np.allclose(new_centers, centers):
        break
    centers = new_centers
