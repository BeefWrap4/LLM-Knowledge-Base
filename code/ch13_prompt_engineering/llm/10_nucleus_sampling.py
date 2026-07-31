# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.3.2 Top-p (Nucleus) Sampling
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 10_nucleus_sampling.py
# expected_runtime: <1s
# expected_output: 打印采样到的 token id 及"核"的大小
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.3.2
# Interview hooks:
# - Top-p 与 Top-k 的本质差异？
# - 为什么 Top-p 更能适应概率分布的"形状"？
# - Top-p 与 Temperature 联合使用时，先后顺序如何影响结果？

import numpy as np


def nucleus_sampling(logits, p: float = 0.9):
    """
    Top-p (Nucleus) Sampling 原理演示

    Args:
        logits: 模型输出的原始分数 [vocab_size]
        p: 累积概率阈值（通常 0.85-0.95）
    """
    if not 0 < p <= 1:
        raise ValueError("p 必须在 (0, 1] 范围内")

    # 1. 计算概率分布；减去最大值避免较大 logits 触发 exp 溢出
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits)

    # 2. 按概率降序排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # 3. 累积概率，找到核
    cumsum = np.cumsum(sorted_probs)
    nucleus_size = min(int(np.searchsorted(cumsum, p, side="left")) + 1, len(sorted_probs))

    # 4. 只在核内重新归一化并采样
    nucleus_probs = sorted_probs[:nucleus_size]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    nucleus_indices = sorted_indices[:nucleus_size]

    # 5. 采样
    chosen = np.random.choice(nucleus_indices, p=nucleus_probs)
    return chosen, nucleus_size


if __name__ == "__main__":
    np.random.seed(42)
    # 模拟一个 vocab_size=10 的 logits 分布
    logits = np.array([2.0, 3.5, 1.0, 4.2, 0.5, 3.9, 1.8, 0.3, 2.7, 1.5])

    for p in [0.5, 0.9, 0.99]:
        chosen, n_size = nucleus_sampling(logits, p=p)
        print(f"p={p:.2f} → 核大小={n_size}, 采样 token id={chosen}")
    print("OK")
