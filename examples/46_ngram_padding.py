"""n-gram 特征与序列截断/补齐。"""
from __future__ import annotations


def make_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def pad_sequences(
    sequences: list[list[int]],
    max_len: int,
    padding: str = "post",
    truncating: str = "post",
    pad_value: int = 0,
) -> list[list[int]]:
    output = []
    for sequence in sequences:
        if truncating == "post":
            row = sequence[:max_len]
        else:
            row = sequence[-max_len:]
        pads = [pad_value] * (max_len - len(row))
        row = row + pads if padding == "post" else pads + row
        output.append(row)
    return output


tokens = ["是谁", "敲动", "我心", "让", "夜色", "明亮"]
print("2-gram:", make_ngrams(tokens, 2))
print("3-gram:", make_ngrams(tokens, 3))

sequences = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1], [2, 32, 1, 23, 1]]
print("post padding/truncating:")
for row in pad_sequences(sequences, max_len=10):
    print(row)
