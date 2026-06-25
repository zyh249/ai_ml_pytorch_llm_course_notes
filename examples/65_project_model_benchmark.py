"""把项目文档中记录的指标整理成可比较表。

这些数字来自不同页面、不同划分和不同硬件，只用于复盘，不应视为严格排行榜。
"""
records = [
    ("TF-IDF + RandomForest", 0.8248, "local"),
    ("FastText char default", 0.8761, "local"),
    ("FastText char autotune", 0.9165, "local"),
    ("BERT classifier", 0.9364, "local"),
    ("LLM API snapshot", 0.6908, "remote"),
    ("Distilled BiLSTM snapshot", 0.9125, "local"),
]
for name, score, mode in sorted(records, key=lambda row: row[1], reverse=True):
    print(f"{name:<30} {score:>7.2%}  {mode}")
