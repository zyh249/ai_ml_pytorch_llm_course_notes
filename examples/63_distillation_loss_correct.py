"""知识蒸馏损失：真实标签交叉熵 + 教师软分布 KL 散度。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    hard_loss = F.cross_entropy(student_logits, labels)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
        log_target=False,
    ) * temperature**2
    return alpha * soft_loss + (1 - alpha) * hard_loss


student = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
teacher = torch.tensor([[4.0, 2.0, 0.0]])
labels = torch.tensor([0])
loss = distillation_loss(student, teacher, labels)
loss.backward()
print("loss =", round(loss.item(), 4))
print("student grad =", student.grad)
