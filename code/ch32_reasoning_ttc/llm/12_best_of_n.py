# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.best_of_n
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 12_best_of_n.py
# expected_runtime: <1s
# expected_output: 不同 N 下的教学模拟结果（不是 benchmark）
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. 为什么 Best-of-N 没有脱离任务难度与 verifier 的通用 scaling law？
#   2. verifier 饱和现象：N 多大时再采样无收益？
#   3. BoN 与 Self-Consistency 的本质区别？哪个用 verifier？
"""Best-of-N: 采样 N 个回答，verifier 选最优。

Snell et al. (2024) 发现推理时计算方法的效果随题目难度显著变化；compute-optimal
策略在论文设置下比 Best-of-N 基线更高效。下面只是带随机种子的教学模拟，不复现
论文结果，也不提供可外推的准确率或成本。
Source: https://arxiv.org/abs/2408.03314
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Sample:
    answer: str
    verifier_score: float
    is_correct: bool


def mock_generate(question: str, n: int) -> list[str]:
    """Mock LLM 采样：用标签构造可检查的候选，仅供演示。"""
    candidates = []
    for i in range(n):
        label = "correct" if random.random() < 0.25 else "wrong"
        candidates.append(f"{label}_answer_{i}_to_{question[:10]}")
    return candidates


def mock_verifier(question: str, answer: str) -> float:
    """Mock verifier：正确候选平均分更高，但保留重叠和误判。"""
    del question
    base = 0.35 if "wrong" in answer else 0.65
    return base + random.uniform(-0.35, 0.35)


def mock_correctness(question: str, answer: str) -> bool:
    return "correct" in answer or "right" in answer


def best_of_n(question: str, n: int) -> Sample:
    samples = mock_generate(question, n)
    scored = []
    for a in samples:
        s = mock_verifier(question, a)
        # 把"正确性"作为最终判定
        scored.append(Sample(a, s, is_correct=mock_correctness(question, a)))
    best = max(scored, key=lambda s: s.verifier_score)
    return best


def scaling_simulation(n_max: int = 64, n_trials: int = 1000) -> dict[int, float]:
    """估计这个合成生成器/verifier 下的选择准确率。"""
    results = {}
    for n in (1, 4, 16, 64):
        if n > n_max:
            continue
        n_correct = 0
        for _ in range(n_trials):
            best = best_of_n("Q", n)
            if best.is_correct:
                n_correct += 1
        results[n] = n_correct / n_trials
    return results


def main() -> None:
    random.seed(0)

    print("=== Best-of-N 单次执行 ===")
    for n in (1, 4, 16):
        s = best_of_n("What is 2+2?", n)
        print(
            f"  N={n:>3}  best_answer={s.answer!r}  verifier={s.verifier_score:.3f}  correct={s.is_correct}"
        )

    print("\n=== 合成 BoN 模拟（1000 trials；不是论文 benchmark）===")
    acc = scaling_simulation()
    print(f"  {'N':>4} | {'selection accuracy':>18} | {'sample units':>12}")
    for n, accuracy in acc.items():
        print(f"  {n:>4} | {accuracy:>17.1%}  | {n:>12}")

    print("\n=== 采样量-选择准确率 trade-off（成本单位仅代表采样次数）===")
    for n, accuracy in acc.items():
        print(f"  N={n:>3}  acc={accuracy:.1%}  sample_units={n}")

    # 与 Self-Consistency 区别
    print("\n=== BoN vs Self-Consistency ===")
    print("  BoN  : 用 verifier 选最优 (reward model)")
    print("  SC   : 用 majority vote 选众数 (无 verifier)")
    print("  组合 : 先 SC 缩到 top-k, 再 RM 重排")
    print("OK")


if __name__ == "__main__":
    main()
