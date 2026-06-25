"""BERT 风格输出如何连接分类头和序列标注头。"""
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit('请先安装 PyTorch：pip install torch') from exc

batch, seq_len, hidden, num_classes, num_tags = 4, 10, 768, 3, 9
encoder_output = torch.randn(batch, seq_len, hidden)

# 文本分类：使用 [CLS] 位置表征。
classification_head = nn.Linear(hidden, num_classes)
cls_logits = classification_head(encoder_output[:, 0, :])

# 实体识别：对每个 token 分别分类。
ner_head = nn.Linear(hidden, num_tags)
ner_logits = ner_head(encoder_output)

print('classification logits:', cls_logits.shape)  # [B, C]
print('NER logits:', ner_logits.shape)             # [B, L, tags]
