"""把 text\tlabel 转成 FastText 监督分类格式。"""
from __future__ import annotations

from pathlib import Path
import jieba

INPUT = Path("train.txt")
OUTPUT = Path("train_fasttext_char.txt")
USE_CHAR = True

with INPUT.open("r", encoding="utf-8") as src, OUTPUT.open("w", encoding="utf-8") as dst:
    for line_no, line in enumerate(src, 1):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            text, label = line.rsplit("\t", 1)
        except ValueError:
            print(f"跳过格式错误行 {line_no}")
            continue
        tokens = list(text) if USE_CHAR else jieba.lcut(text)
        clean = " ".join(token for token in tokens if token.strip())
        dst.write(f"__label__{label} {clean}\n")

print("已生成:", OUTPUT)
print("训练示例：")
print("import fasttext")
print("model = fasttext.train_supervised(input='train_fasttext_char.txt')")
