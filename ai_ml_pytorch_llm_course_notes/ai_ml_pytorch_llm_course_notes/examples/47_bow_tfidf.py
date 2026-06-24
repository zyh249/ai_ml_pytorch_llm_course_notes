"""使用 scikit-learn 比较词频编码与 TF-IDF。"""
from __future__ import annotations

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except ImportError as exc:
    raise SystemExit("请先安装 scikit-learn：pip install scikit-learn") from exc

documents = [
    "房间 干净 服务 很好",
    "服务 一般 早餐 很少",
    "位置 很好 交通 方便",
]

count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
count_matrix = count_vectorizer.fit_transform(documents)
print("词表:", count_vectorizer.get_feature_names_out().tolist())
print("词频矩阵:\n", count_matrix.toarray())

tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("\nTF-IDF 矩阵:\n", tfidf_matrix.toarray().round(3))
