"""把 text\\tlabel 数据转换为 FastText 监督学习格式。"""
from __future__ import annotations

import re

LABELS = ["finance", "realty", "stocks", "education", "science", "society", "politics", "sports", "game", "entertainment"]


def to_fasttext_line(text: str, label_id: int, char_level: bool = True) -> str:
    text = re.sub(r"\s+", "", text.replace("：", ""))
    tokens = list(text) if char_level else text.split()
    return f"__label__{LABELS[label_id]} " + " ".join(tokens)


print(to_fasttext_line("中华女子学院：本科层次仅1专业招男生", 3, char_level=True))
