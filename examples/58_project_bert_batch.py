"""BERT 文本分类批处理：固定最大长度、动态张量化和注意力掩码。"""
from __future__ import annotations

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as exc:
    raise SystemExit("请安装：pip install torch transformers") from exc

MODEL_NAME = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
texts = ["中华女子学院本科层次仅1专业招男生", "今日大盘上涨"]
batch = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
for key, value in batch.items():
    print(key, tuple(value.shape))
    print(value)
