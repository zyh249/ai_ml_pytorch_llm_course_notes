"""从 token 序列构造 CBOW 与 Skip-gram 训练样本。"""
from __future__ import annotations


def cbow_pairs(tokens: list[str], window: int = 2):
    for center in range(window, len(tokens) - window):
        context = tokens[center - window:center] + tokens[center + 1:center + window + 1]
        yield context, tokens[center]


def skipgram_pairs(tokens: list[str], window: int = 2):
    for center, center_word in enumerate(tokens):
        left = max(0, center - window)
        right = min(len(tokens), center + window + 1)
        for context_index in range(left, right):
            if context_index != center:
                yield center_word, tokens[context_index]


tokens = "清晨 微风 穿过 安静 窗台 阳光 落下".split()
print("CBOW：上下文 -> 中心词")
for context, target in list(cbow_pairs(tokens))[:4]:
    print(context, "->", target)

print("\nSkip-gram：中心词 -> 上下文词")
for center, context in list(skipgram_pairs(tokens))[:8]:
    print(center, "->", context)
