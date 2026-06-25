"""BERT 文本分类的数据集、collate_fn 与输入形状模板。"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise SystemExit("请安装 transformers：pip install transformers") from exc


class NewsDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.rows[index]


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


rows = [
    ("中华女子学院本科层次仅一专业招男生", 3),
    ("卡佩罗谈德国队比赛", 7),
]
loader = DataLoader(NewsDataset(rows), batch_size=2, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))
for key, value in batch.items():
    print(key, tuple(value.shape))
