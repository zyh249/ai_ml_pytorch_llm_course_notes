"""LLM 分类输出约束与健壮解析，不执行真实 API 请求。"""
from __future__ import annotations

import re

LABELS = {"finance", "realty", "stocks", "education", "science", "society", "politics", "sports", "game", "entertainment", "reject"}


def parse_label(raw: str) -> str:
    normalized = raw.strip().lower()
    match = re.search(r"(?:文本类别|类别)\s*[:：]\s*([a-z_]+)", normalized)
    candidate = match.group(1) if match else normalized.split()[0].strip("`'\".,:：")
    if candidate not in LABELS:
        raise ValueError(f"模型返回了非法类别：{raw!r}")
    return candidate


for output in ["文本类别：stocks", "sports", "类别: reject"]:
    print(output, "->", parse_label(output))
