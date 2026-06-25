"""jieba 三种分词模式、词性标注和一个简单实体识别示例。"""
from __future__ import annotations

try:
    import jieba
    import jieba.posseg as pseg
except ImportError as exc:
    raise SystemExit("请先安装 jieba：pip install jieba") from exc

text = "传智教育是一家上市公司，旗下有黑马程序员品牌。"

print("精确模式:")
print(list(jieba.cut(text, cut_all=False)))

print("\n全模式:")
print(list(jieba.cut(text, cut_all=True)))

print("\n搜索引擎模式:")
print(list(jieba.cut_for_search(text)))

print("\n词性标注:")
for word, flag in pseg.cut("我爱自然语言处理"):
    print(f"{word}/{flag}", end="  ")
print()

# 教学用规则 NER：真实项目通常使用训练好的序列标注模型。
entity_dictionary = {
    "传智教育": "机构",
    "黑马程序员": "品牌/机构",
    "浙江绍兴": "地点",
    "鲁迅": "人物",
}
example = "鲁迅是浙江绍兴人，传智教育旗下有黑马程序员品牌。"
print("\n规则实体识别:")
for entity, entity_type in entity_dictionary.items():
    if entity in example:
        print(entity, "->", entity_type)
