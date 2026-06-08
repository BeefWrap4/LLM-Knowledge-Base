# ---
# chapter: 27
# topic: Best-of-N sampling with reward model
# section: 27.5.2 Best-of-N
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 12_best_of_n.py
# expected_runtime: <1s
# expected_output: 不同 N 下准确率 / 成本对比
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.5.2 + §27.3
# Interview hooks:
#   1. Best-of-N 推理时扩展的 scaling law 是怎样的？
#   2. verifier 饱和现象：N 多大时再采样无收益？
#   3. BoN 与 Self-Consistency 的本质区别？哪个用 verifier？
"""Best-of-N: 采样 N 个回答，verifier 选最优。

Snell et al. 2024 论文: 在 MATH 任务上, N=256 时准确率从 ~50% 提升到 ~90%。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    answer: str
    verifier_score: float
    is_correct: bool


def mock_generate(question: str, n: int) -> list[str]:
    """Mock LLM 采样：返回 N 个候选。"""
    return [f"answer_{i}_to_{question[:10]}" for i in range(n)]


def mock_verifier(question: str, answer: str) -> float:
    """Mock verifier：随机给分，但正确答案概率更高。"""
    base = 0.3 if "wrong" in answer else 0.7
    return base + random.uniform(-0.2, 0.2)


def mock_correctness(question: str, answer: str) -> bool:
    return "correct" in answer or "right" in answer


def best_of_n(question: str, n: int, gt_correct: bool = True) -> Sample:
    samples = mock_generate(question, n)
    scored = []
    for a in samples:
        s = mock_verifier(question, a)
        # 把"正确性"作为最终判定
        scored.append(Sample(a, s, is_correct=mock_correctness(question, a)))
    best = max(scored, key=lambda s: s.verifier_score)
    return best


def scaling_law_simulation(n_max: int = 256, n_trials: int = 1000) -> dict:
    """估计不同 N 下的 BoN 准确率。"""
    results = {}
    for N in (1, 4, 16, 64, 256):
        N = min(N, n_max)
        n_correct = 0
        for _ in range(n_trials):
            best = best_of_n("Q", N, gt_correct=True)
            if best.is_correct:
                n_correct += 1
        results[N] = n_correct / n_trials
    return results


def main() -> None:
    random.seed(0)
    np.random.seed(0)

    print("=== Best-of-N 单次执行 ===")
    for N in (1, 4, 16):
        s = best_of_n("What is 2+2?", N)
        print(
            f"  N={N:>3}  best_answer={s.answer!r}  verifier={s.verifier_score:.3f}  correct={s.is_correct}"
        )

    # Scaling law
    print("\n=== BoN scaling (1000 trials) ===")
    acc = scaling_law_simulation()
    print(f"  {'N':>4} | {'accuracy':>8} | {'cost×':>6}")
    for N, a in acc.items():
        print(f"  {N:>4} | {a:>7.1%}  | {N:>5}x")

    # 成本/准确率
    print("\n=== 成本-准确率 trade-off (假设 o3-mini @ $0.005/call) ===")
    for N, a in acc.items():
        cost = N * 0.005
        # 反推"每 1% 准确率提升的成本"
        print(f"  N={N:>3}  acc={a:.1%}  cost=${cost:.3f}")

    # 与 Self-Consistency 区别
    print("\n=== BoN vs Self-Consistency ===")
    print("  BoN  : 用 verifier 选最优 (reward model)")
    print("  SC   : 用 majority vote 选众数 (无 verifier)")
    print("  组合 : 先 SC 缩到 top-k, 再 RM 重排")


if __name__ == "__main__":
    main()
