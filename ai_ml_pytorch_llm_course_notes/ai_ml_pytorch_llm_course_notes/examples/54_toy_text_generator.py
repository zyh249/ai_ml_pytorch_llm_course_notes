"""一个小型、可离线运行的词级 RNN 文本生成示例。

语料为自定义短句，避免依赖外部数据。训练规模很小，只用于理解流程。
"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

torch.set_num_threads(1)
torch.manual_seed(7)

CORPUS = """
清晨 微风 穿过 窗台
夜色 星光 落在 海面
我们 沿着 小路 慢慢 前行
雨后 天空 格外 明亮
""".strip().split()

vocab = ["<BOS>", "<EOS>"] + sorted(set(CORPUS))
word_to_id = {word: index for index, word in enumerate(vocab)}
id_to_word = dict(enumerate(vocab))
ids = [word_to_id[word] for word in CORPUS]

# 滑动窗口：输入前 4 个词，目标是右移 1 位后的 4 个词。
window = 4
X, Y = [], []
for start in range(len(ids) - window):
    X.append(ids[start:start + window])
    Y.append(ids[start + 1:start + window + 1])
x = torch.tensor(X)
y = torch.tensor(Y)

class ToyGenerator(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 24)
        self.rnn = nn.RNN(24, 32, batch_first=True)
        self.output = nn.Linear(32, vocab_size)

    def forward(self, token_ids, hidden=None):
        vectors = self.embedding(token_ids)
        features, hidden = self.rnn(vectors, hidden)
        return self.output(features), hidden

model = ToyGenerator(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(12):
    logits, _ = model(x)
    loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

start_word = "清晨"
current = torch.tensor([[word_to_id[start_word]]])
hidden = None
generated = [start_word]
for _ in range(12):
    logits, hidden = model(current, hidden)
    next_id = int(logits[0, -1].argmax())
    generated.append(id_to_word[next_id])
    current = torch.tensor([[next_id]])

print(" ".join(generated))
