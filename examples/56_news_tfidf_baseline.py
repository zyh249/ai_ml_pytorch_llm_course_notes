"""投满分项目：TF-IDF + 经典分类器基线。

示例优先使用 LinearSVC；它通常比随机森林更适合高维稀疏文本。
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

texts = [
    "中华女子学院本科层次专业招生", "考研英语复习全指南", "高校发布招生简章",
    "A股净流入多只股票上涨", "上市公司发布季度财报", "基金重仓股表现活跃",
    "新盘两居准现房优惠", "住宅成交量环比增长", "城市发布购房新政",
    "国足完成赛前训练", "球队客场击败对手", "联赛积分榜发生变化",
]
labels = ["education"] * 3 + ["stocks"] * 3 + ["realty"] * 3 + ["sports"] * 3

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42, stratify=labels
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
    ("classifier", LinearSVC()),
])
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(classification_report(y_test, pred, zero_division=0))
print(model.predict(["中国人民公安大学硕士研究生目录"])[0])
