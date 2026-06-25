"""LLM 闭集分类：提示词构造、结构化输出解析与拒识校验。

脚本不发送网络请求，便于先测试提示词和解析逻辑。
"""
from __future__ import annotations

import json

ALLOWED_LABELS = {
    "finance", "realty", "stocks", "education", "science",
    "society", "politics", "sports", "game", "entertainment", "reject",
}


def build_prompt(text: str) -> str:
    labels = ", ".join(sorted(ALLOWED_LABELS))
    return f"""你是新闻分类器。只能从以下标签中选择一个：{labels}。
输出严格 JSON，不要解释：{{"label": "...", "confidence": 0.0}}
若文本不属于任何类别或信息不足，label 必须为 reject。
待分类文本：{text}"""


def parse_response(raw: str) -> tuple[str, float]:
    data = json.loads(raw)
    label = data.get("label")
    confidence = float(data.get("confidence", 0.0))
    if label not in ALLOWED_LABELS:
        raise ValueError(f"模型返回未知标签：{label!r}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence 必须在 0~1")
    return label, confidence


print(build_prompt("今日A股净流入520亿，多数股票上涨"))
print(parse_response('{"label":"stocks","confidence":0.92}'))
