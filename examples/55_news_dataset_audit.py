"""投满分项目：TSV 新闻数据集审计工具。

用法：
    python 55_news_dataset_audit.py train.txt class.txt
没有参数时会用内置小样本演示。
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


def read_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError("class.txt 不能为空")
    return classes


def read_tsv(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="	")
        for line_no, row in enumerate(reader, 1):
            if len(row) != 2:
                raise ValueError(f"第 {line_no} 行不是两列：{row!r}")
            text, label_text = row
            if not text.strip():
                raise ValueError(f"第 {line_no} 行文本为空")
            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(f"第 {line_no} 行标签不是整数：{label_text!r}") from exc
            rows.append((text.strip(), label))
    return rows


def audit(rows: list[tuple[str, int]], classes: list[str]) -> None:
    label_counts = Counter(label for _, label in rows)
    invalid = sorted(label for label in label_counts if not 0 <= label < len(classes))
    if invalid:
        raise ValueError(f"发现越界标签：{invalid}")

    lengths = [len(text) for text, _ in rows]
    duplicate_count = len(rows) - len(set(rows))
    print(f"样本数：{len(rows):,}")
    print(f"类别数：{len(classes)}")
    print(f"重复样本：{duplicate_count:,}")
    print(
        "文本长度：",
        f"min={min(lengths)}, mean={statistics.mean(lengths):.2f}, "
        f"median={statistics.median(lengths):.2f}, max={max(lengths)}",
    )
    print("
类别分布：")
    for label_id, class_name in enumerate(classes):
        count = label_counts.get(label_id, 0)
        ratio = count / len(rows) if rows else 0
        print(f"  {label_id:>2} {class_name:<15} {count:>8,}  {ratio:>7.2%}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        data_rows = read_tsv(Path(sys.argv[1]))
        class_names = read_classes(Path(sys.argv[2]))
    else:
        class_names = ["finance", "realty", "stocks", "education"]
        data_rows = [
            ("中华女子学院本科层次仅一专业招男生", 3),
            ("今日A股成交额放大，多只股票上涨", 2),
            ("新盘推出两居准现房优惠", 1),
            ("银行下调部分理财产品费率", 0),
        ]
    audit(data_rows, class_names)
