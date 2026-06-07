# ---
# chapter: 27
# topic: GRPO advantage estimation and group size effect
# section: 27.6.2 RL 阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 06_grpo_advantage.py
# expected_runtime: <2s
# expected_output: 不同 G 下 advantage 方差对比
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.6.2
# Interview hooks:
#   1. 组大小 G 对优势估计方差的影响？为何 G 越大训练越稳？
#   2. 不用 Critic 的话，深套话状态时怎么估计 advantage？(token-level GRPO)
#   3. 当组内所有回答奖励相同时优势是多少？此时梯度？
"""GRPO 优势估计的统计性质。

可视化：组大小 G 对优势估计方差的影响，以及 baseline 漂移时的稳定性。
"""
from __future__ import annotations

import numpy as np


def grpo_advantages(rewards: np.ndarray) -> np.ndarray:
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def estimate_advantage_variance(
    true_mean: float, true_std: float, group_size: int, n_trials: int = 5000
) -> float:
    """蒙特卡洛估计：在给定真实分布下，组大小 G 估计优势的方差。"""
    advantages_var = []
    for _ in range(n_trials):
        r = np.random.normal(true_mean, true_std, size=group_size)
        a = grpo_advantages(r)
        advantages_var.append(a.var())
    return float(np.mean(advantages_var))


def main() -> None:
    print("=== GRPO 优势估计方差 vs 组大小 G ===\n")
    print(f"{'G':>4} | {'优势方差(估计)':>16} | {'归一化':>8}")
    print("-" * 38)
    for G in (2, 4, 8, 16, 32, 64, 128):
        var = estimate_advantage_variance(0.5, 1.0, G)
        # 归一化到 G=2 为基准
        norm = var / estimate_advantage_variance(0.5, 1.0, 2)
        print(f"{G:>4} | {var:>16.4f} | {norm:>7.2f}x")

    # 演示：组内奖励全相等 → 优势全为 0 → 无梯度
    print("\n=== 边界情况 ===")
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    a = grpo_advantages(rewards)
    print(f"全相同奖励 {rewards} → 优势 {a} (无信号，policy 不更新)")

    # 演示：一个明显最优 + 其余平均
    rewards = np.array([1.0, 0.1, 0.1, 0.1])
    a = grpo_advantages(rewards)
    print(f"1 优 3 差    {rewards} → 优势 {a.round(2)} (优样本正优势被放大)")

    # 演示：G=2 时方差高
    rng = np.random.default_rng(42)
    n = 200
    a2, a16 = [], []
    for _ in range(n):
        a2.append(grpo_advantages(rng.normal(0, 1, 2)).var())
        a16.append(grpo_advantages(rng.normal(0, 1, 16)).var())
    print(f"\nG=2  优势方差 stdev: {np.std(a2):.3f}")
    print(f"G=16 优势方差 stdev: {np.std(a16):.3f}  (更稳)")


if __name__ == "__main__":
    main()
