"""NewsCompass 数据集 EDA：读取 text\tlabel，统计类别与文本长度。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import pandas as pd

DATA_PATH = Path("train.txt")
if not DATA_PATH.exists():
    raise SystemExit("请把 train.txt 放到当前目录，格式为：文本<TAB>标签")

df = pd.read_csv(DATA_PATH, sep="\t", names=["text", "label"], dtype={"text": str})
df["text"] = df["text"].fillna("")
df["char_len"] = df["text"].str.len()

print(df.head())
print("样本数:", len(df))
print("标签分布:", Counter(df["label"]))
print("文本长度统计:")
print(df["char_len"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
print("重复文本数:", int(df["text"].duplicated().sum()))
print("空文本数:", int(df["text"].eq("").sum()))
