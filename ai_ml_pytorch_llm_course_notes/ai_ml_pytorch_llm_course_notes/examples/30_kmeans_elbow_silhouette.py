"""KMeans：同时计算 SSE 和轮廓系数。"""
try:
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
except ImportError as exc:
    raise SystemExit('请先安装 scikit-learn：pip install scikit-learn') from exc

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.65, random_state=7)
for k in range(2, 8):
    model = KMeans(n_clusters=k, n_init=10, random_state=7)
    labels = model.fit_predict(X)
    print(
        f'k={k}, SSE={model.inertia_:.2f}, '
        f'silhouette={silhouette_score(X, labels):.4f}'
    )
