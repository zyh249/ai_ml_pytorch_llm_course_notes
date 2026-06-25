"""把项目讲义中的模型结果整理成可审计表格。"""
from __future__ import annotations

rows = [
    ("RandomForest + TF-IDF", 82.48, "10k test"),
    ("FastText char default", 87.61, "10k test"),
    ("FastText char autotune", 91.65, "10k test"),
    ("FastText word default", 90.79, "10k test; 取日志值0.9079"),
    ("FastText word autotune", 90.41, "10k test"),
    ("BERT classifier", 93.64, "项目报告值"),
    ("DeepSeek API prompt", 69.08, "511 samples; 非同一评估条件"),
]
for name, score, note in sorted(rows, key=lambda r: r[1], reverse=True):
    print(f"{name:<30} {score:>6.2f}%  {note}")
