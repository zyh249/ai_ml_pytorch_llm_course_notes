"""Scaled Dot-Product Attention：从 Q、K、V 到注意力输出。"""
try:
    import torch
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

def attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights

# 3 个 token，每个 token 的向量维度为 4
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # 这家
    [0.0, 1.0, 0.0, 1.0],  # 苹果/拉面等歧义词
    [1.0, 1.0, 0.0, 0.0],  # 太棒了
])
q = k = v = x
out, weights = attention(q, k, v)
print("attention weights:")
print(weights.round(decimals=3))
print("context-aware output:")
print(out.round(decimals=3))

# Decoder 常用的因果 mask：当前位置不能看未来词
causal_mask = torch.tril(torch.ones(3, 3))
_, masked_weights = attention(q, k, v, causal_mask)
print("masked weights:")
print(masked_weights.round(decimals=3))
