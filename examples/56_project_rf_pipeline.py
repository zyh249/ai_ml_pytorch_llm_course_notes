"""Jieba + TF-IDF + RandomForest 的可复用 Pipeline。"""
from __future__ import annotations

try:
    import jieba
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
except ImportError as exc:
    raise SystemExit("请安装：pip install jieba scikit-learn") from exc


def tokenize(text: str) -> list[str]:
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


model = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_features=30_000)),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
])

x_train = ["股市今日大涨", "大学招生简章", "足球联赛开幕", "基金净值上涨", "研究生考试指南", "篮球决赛"]
y_train = [2, 3, 7, 0, 3, 7]
x_test = ["今日股票上涨", "考研复习计划", "世界杯比赛"]
y_test = [2, 3, 7]
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(classification_report(y_test, pred, zero_division=0))
