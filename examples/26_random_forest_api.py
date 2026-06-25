"""scikit-learn 随机森林分类完整调用示例。"""
try:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError as exc:
    raise SystemExit('请先安装 scikit-learn：pip install scikit-learn') from exc

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('test accuracy =', accuracy_score(y_test, pred))
print('OOB score =', model.oob_score_)
print('feature importance =', model.feature_importances_)
