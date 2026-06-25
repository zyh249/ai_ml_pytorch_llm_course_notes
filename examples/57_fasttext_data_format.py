"""把 text<TAB>label 转换为 FastText supervised 格式。"""
from __future__ import annotations

import re
from typing import Iterable

CLASSES = [
    "finance", "realty", "stocks", "education", "science",
    "society", "politics", "sports", "game", "entertainment",
]


def char_tokens(text: str) -> list[str]:
    return [char for char in text.strip() if not char.isspace()]


def simple_word_tokens(text: str) -> list[str]:
    # 真正中文项目可替换为 jieba.lcut(text)。
    return [token for token in re.split(r"[，。！？、\s]+", text.strip()) if token]


def to_fasttext(text: str, label_id: int, *, char_level: bool = True) -> str:
    if not 0 <= label_id < len(CLASSES):
        raise ValueError(f"label_id 越界：{label_id}")
    tokens = char_tokens(text) if char_level else simple_word_tokens(text)
    return f"__label__{CLASSES[label_id]} " + " ".join(tokens)


samples: Iterable[tuple[str, int]] = [
    ("三羊资产清盘事起ITAT", 0),
    ("卡佩罗谈德国队比赛", 7),
]
for text, label in samples:
    print(to_fasttext(text, label, char_level=True))
