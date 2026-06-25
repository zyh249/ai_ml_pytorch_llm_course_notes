"""正确的软标签蒸馏损失：交叉熵 + KL 散度。"""
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("请安装：pip install torch") from exc


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
    hard_loss = F.cross_entropy(student_logits, labels)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2
    return alpha * soft_loss + (1 - alpha) * hard_loss

student = torch.randn(4, 10, requires_grad=True)
teacher = torch.randn(4, 10)
labels = torch.tensor([1, 2, 3, 4])
loss = distillation_loss(student, teacher, labels)
loss.backward()
print("loss =", float(loss.detach()))
