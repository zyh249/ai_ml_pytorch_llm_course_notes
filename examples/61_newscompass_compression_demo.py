"""BERT 分类模型的动态量化、蒸馏损失与剪枝核心写法。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import prune


def dynamic_int8(model: nn.Module) -> nn.Module:
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    hard = nn.functional.cross_entropy(student_logits, labels)
    soft = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=-1),
        nn.functional.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    return alpha * soft + (1 - alpha) * hard


def prune_linear_layers(model: nn.Module, amount=0.30) -> None:
    parameters = [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    for module, name in parameters:
        prune.remove(module, name)  # 固化剪枝结果，移除重参数化钩子
