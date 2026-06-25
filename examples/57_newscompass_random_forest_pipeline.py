"""Jieba + TF-IDF + RandomForest 的可复用 Pipeline。"""
from __future__ import annotations

import joblib
import jieba
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def cut(text: str) -> str:
    return " ".join(jieba.lcut(str(text))[:30])


df = pd.read_csv("train.txt", sep="\t", names=["text", "label"])
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=22, stratify=df["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=cut, min_df=2, max_features=100_000)),
    ("model", RandomForestClassifier(
        n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=22
    )),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test), digits=4))
joblib.dump(pipeline, "newscompass_rf.joblib")
