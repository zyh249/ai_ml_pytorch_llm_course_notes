"""语言模型常见评价指标的简化计算示例。"""
import math
from collections import Counter

# Perplexity：已知真实 token 的负对数似然。
true_token_probabilities = [0.8, 0.5, 0.25, 0.9]
mean_nll = -sum(math.log(p) for p in true_token_probabilities) / len(true_token_probabilities)
ppl = math.exp(mean_nll)
print('perplexity =', round(ppl, 4))

# 简化的 unigram precision/recall，帮助理解 BLEU/ROUGE 的方向。
reference = '模型 使用 注意力 处理 上下文'.split()
candidate = '模型 使用 注意力 理解 上下文'.split()
ref_count, cand_count = Counter(reference), Counter(candidate)
overlap = sum((ref_count & cand_count).values())
precision = overlap / len(candidate)  # BLEU 更偏精确率方向
recall = overlap / len(reference)     # ROUGE 更偏召回率方向
print('unigram precision =', round(precision, 4))
print('unigram recall =', round(recall, 4))
