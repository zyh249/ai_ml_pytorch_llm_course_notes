"""GPT 生成中的 temperature、top-k 和随机采样。"""
import math
import random

vocab = ['机器', '学习', '很', '有趣', '困难']
logits = [2.4, 1.8, 1.0, 0.7, -0.3]


def softmax(xs, temperature=1.0):
    scaled = [x / temperature for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def top_k_filter(tokens, probs, k):
    pairs = sorted(zip(tokens, probs), key=lambda x: x[1], reverse=True)[:k]
    norm = sum(p for _, p in pairs)
    return [(t, p / norm) for t, p in pairs]

for temperature in (0.4, 1.0, 1.8):
    probs = softmax(logits, temperature)
    candidates = top_k_filter(vocab, probs, k=3)
    token = random.choices(
        [t for t, _ in candidates],
        weights=[p for _, p in candidates],
        k=1,
    )[0]
    print(f'T={temperature}:', candidates, 'sample=', token)
