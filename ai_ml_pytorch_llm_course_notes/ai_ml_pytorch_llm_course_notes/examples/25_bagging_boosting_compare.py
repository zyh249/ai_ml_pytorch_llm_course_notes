"""Bagging 与 Boosting 的流程差异示意。"""
from collections import Counter

# Bagging：各弱学习器独立、平权投票
bagging_predictions = [1, 1, 0, 1, 0, 1, 1]
print('Bagging vote:', Counter(bagging_predictions).most_common(1)[0][0])

# Boosting：串行叠加，每个学习器带权重
boosting_predictions = [1, 0, 1]
learner_weights = [0.3, 0.8, 1.2]
score = sum((1 if pred == 1 else -1) * w for pred, w in zip(boosting_predictions, learner_weights))
print('Boosting weighted score:', score)
print('Boosting result:', 1 if score >= 0 else 0)
