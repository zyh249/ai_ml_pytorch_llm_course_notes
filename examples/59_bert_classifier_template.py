"""BERT + 线性分类头：更稳健的训练骨架。"""
from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import AutoModel
except ImportError as exc:
    raise SystemExit("请安装 transformers：pip install transformers") from exc


class BertNewsClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-chinese", num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = output.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_vector))


# 训练顺序：zero_grad -> forward -> CrossEntropyLoss -> backward -> clip -> step
