"""bert-base-chinese + 线性分类头的最小实现。"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-chinese"
NUM_CLASSES = 10

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class BertClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_CLASSES)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))


texts = ["中华女子学院：本科层次仅1专业招男生", "体验2D巅峰游戏"]
batch = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
model = BertClassifier()
logits = model(batch["input_ids"], batch["attention_mask"])
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("logits:", logits.shape)
