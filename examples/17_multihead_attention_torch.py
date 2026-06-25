"""PyTorch 多头注意力：观察输入、权重和输出形状。"""
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit('请先安装 PyTorch：pip install torch') from exc

torch.manual_seed(0)
batch_size, seq_len, d_model, n_heads = 2, 5, 12, 3
x = torch.randn(batch_size, seq_len, d_model)

mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=n_heads,
    batch_first=True,
)
output, weights = mha(x, x, x, need_weights=True, average_attn_weights=False)
print('input :', x.shape)       # [B, L, d_model]
print('output:', output.shape)  # [B, L, d_model]
print('weights:', weights.shape) # [B, heads, L, L]
print('head-0 weights of sample-0:\n', weights[0, 0].round(decimals=3))
