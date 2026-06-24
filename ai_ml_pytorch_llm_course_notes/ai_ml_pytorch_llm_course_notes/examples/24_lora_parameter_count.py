"""LoRA 参数量计算：ΔW = A @ B。"""

def lora_stats(n: int, rank: int):
    full = n * n
    lora = n * rank + rank * n
    return full, lora, lora / full

for n, rank in [(768, 8), (4096, 8), (4096, 64)]:
    full, lora, ratio = lora_stats(n, rank)
    print(f'n={n}, rank={rank}: full={full:,}, LoRA={lora:,}, ratio={ratio:.4%}')
