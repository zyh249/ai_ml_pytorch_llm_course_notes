"""回译增强模板。

真实项目中 translator 通常封装翻译 API 或本地翻译模型。
本示例只定义流程，不发送任何网络请求。
"""
from __future__ import annotations

from collections.abc import Callable


def back_translate(
    text: str,
    translate: Callable[[str, str, str], str],
    pivot_language: str = "en",
) -> str:
    pivot = translate(text, "zh", pivot_language)
    augmented = translate(pivot, pivot_language, "zh")
    return augmented


def mock_translate(text: str, source: str, target: str) -> str:
    # 仅用于演示接口；真实翻译应替换这里。
    table = {
        ("房间很干净，服务也很好。", "zh", "en"): "The room is clean and the service is good.",
        ("The room is clean and the service is good.", "en", "zh"): "客房整洁，服务也不错。",
    }
    return table.get((text, source, target), text)


original = "房间很干净，服务也很好。"
augmented = back_translate(original, mock_translate)
print("原句:", original)
print("回译:", augmented)
print("标签保持不变，但仍需人工或规则抽检语义是否漂移。")
