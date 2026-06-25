"""NewsCompass 数据集 EDA：标签分布与文本长度统计。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics


def load_tsv(path: str | Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                text, label = line.rsplit("\t", 1)
                rows.append((text, int(label)))
            except ValueError as exc:
                raise ValueError(f"第 {line_no} 行不是 text\\tlabel 格式") from exc
    return rows


def summarize(rows: list[tuple[str, int]]) -> None:
    labels = Counter(label for _, label in rows)
    lengths = [len(text) for text, _ in rows]
    print("样本数:", len(rows))
    print("标签分布:", dict(sorted(labels.items())))
    print("长度 min/mean/median/max:", min(lengths), round(statistics.mean(lengths), 2), statistics.median(lengths), max(lengths))


if __name__ == "__main__":
    demo = [("中华女子学院本科层次仅1专业招男生", 3), ("今日大盘上涨", 2), ("足球比赛精彩回顾", 7)]
    summarize(demo)
