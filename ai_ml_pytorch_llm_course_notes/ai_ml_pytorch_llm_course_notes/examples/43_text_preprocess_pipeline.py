"""一个可运行的中文文本预处理最小流水线。

步骤：清洗 -> 分词 -> 建词表 -> 数字化 -> 截断/补齐。
安装 jieba 后会使用 jieba；没有安装时退化为逐字切分。
"""
from __future__ import annotations

import re
from collections import Counter

try:
    import jieba
except ImportError:
    jieba = None

TEXTS = [
    "房间干净，前台服务很好！",
    "早餐种类少，隔音也不太好。",
    "位置方便，下次还会入住。",
]


def clean_text(text: str) -> str:
    """保留中文、英文、数字和常用标点，压缩空白。"""
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9，。！？、 ]", "", text)


def tokenize(text: str) -> list[str]:
    if jieba is not None:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    return [char for char in text if not char.isspace()]


def build_vocab(tokenized: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    counter = Counter(token for sentence in tokenized for token in sentence)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["<UNK>"]
    return [vocab.get(token, unk) for token in tokens]


def pad_or_truncate(ids: list[int], max_len: int, pad_id: int = 0) -> list[int]:
    ids = ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


cleaned = [clean_text(text) for text in TEXTS]
tokenized = [tokenize(text) for text in cleaned]
vocab = build_vocab(tokenized)
encoded = [encode(tokens, vocab) for tokens in tokenized]
batch = [pad_or_truncate(ids, max_len=10) for ids in encoded]

print("cleaned:", cleaned)
print("tokenized:", tokenized)
print("vocab:", vocab)
print("padded batch:")
for row in batch:
    print(row)
