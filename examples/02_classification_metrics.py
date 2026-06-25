"""二分类指标：accuracy、precision、recall、F1 的手工计算。"""
y_true = [1, 0, 1, 1, 1, 0]
y_pred = [1, 1, 0, 1, 0, 0]

tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

accuracy = (tp + tn) / len(y_true)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"accuracy = {accuracy:.2f}")
print(f"precision = {precision:.2f}")
print(f"recall    = {recall:.2f}")
print(f"f1        = {f1:.2f}")
