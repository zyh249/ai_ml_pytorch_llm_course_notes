"""BERT + Linear 分类头的最小结构。"""
from __future__ import annotations

try:
    import torch
    from torch import nn
    from transformers import AutoModel
except ImportError as exc:
    raise SystemExit("请安装：pip install torch transformers") from exc


class BertClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-chinese", num_classes: int = 10) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]  # [CLS]
        return self.classifier(self.dropout(pooled))
