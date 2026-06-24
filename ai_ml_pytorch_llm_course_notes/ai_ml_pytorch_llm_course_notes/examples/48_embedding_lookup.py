"""PyTorch nn.Embedding：词索引查表得到密集向量。"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.manual_seed(7)
vocab = {"<PAD>": 0, "<UNK>": 1, "房间": 2, "干净": 3, "服务": 4, "很好": 5}
ids = torch.tensor([[2, 3, 4, 5, 0]])  # [batch=1, seq_len=5]

embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=4,
    padding_idx=vocab["<PAD>"],
)
vectors = embedding(ids)

print("embedding weight shape:", embedding.weight.shape)
print("input ids shape:", ids.shape)
print("output vectors shape:", vectors.shape)
print(vectors.round(decimals=3))
