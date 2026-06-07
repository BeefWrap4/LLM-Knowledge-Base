# ---
# chapter: 27
# topic: Test-Time Compute Scaling Law (Snell 2024)
# section: 27.3.2 扩展定律
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 14_ttc_scaling_law.py
# expected_runtime: <1s
# expected_output: 打印 accuracy vs compute 的 scaling 曲线
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3.2 + §27.8 Q1
# Interview hooks:
#   1. TTC scaling law 的数学形式？和预训练 scaling law 区别？
#   2. 何时 BoN > CoT-extension > MCTS-PRM？
#   3. verifier 饱和点 vs 模型能力饱和点？
"""Test-Time Compute Scaling 模拟。

Snell et al. 2024 核心结论 (MATH 任务):
  accuracy ≈ 1 - (c0 / compute)^α   α ≈ 0.3-0.5
  即 256× compute ≈ +40% 准确率。
"""
from __future__ import annotations

import math

import numpy as np


def accuracy_at_compute(
    compute: float, base_acc: float = 0.5, alpha: float = 0.4, c0: float = 1.0
) -> float:
    """log-linear scaling: log(1-acc) = α·log(c0) - α·log(compute)
    → 简化: acc = 1 - (c0/compute)^α
    """
    if compute <= 0:
        return 0.0
    return 1.0 - (c0 / compute) ** alpha * (1 - base_acc)


def cost_efficiency(compute: float, model_cost_per_token: float = 1e-6) -> float:
    """每 1% 准确率提升需要的额外 cost。"""
    return compute * model_cost_per_token


def main() -> None:
    # 不同 scaling 策略对应的"compute"
    strategies = {
        "CoT-base (1×)": 1,
        "CoT-extended (4×)": 4,
        "Self-Consistency N=8": 8,
        "BoN N=64": 64,
        "BoN N=256": 256,
        "MCTS+PRM": 128,
    }

    print("=== Test-Time Compute Scaling (MATH-style) ===\n")
    print(f"  {'Strategy':<22} {'Compute':>8} {'Accuracy':>9} {'Δ acc':>7}")
    print("  " + "-" * 50)
    base = accuracy_at_compute(1.0)
    for name, c in strategies.items():
        acc = accuracy_at_compute(c)
        delta = acc - base
        bar = "█" * int(acc * 40)
        print(f"  {name:<22} {c:>7}× {acc:>8.1%} {delta:>+6.1%}  {bar}")

    # 三策略对比: 同样 compute 下哪种最准?
    print("\n=== 同样 compute=64 下三策略对比 ===")
    compute = 64
    a_bon = accuracy_at_compute(compute)                       # 全 BoN
    a_sc = accuracy_at_compute(compute * 0.7) * 0.97           # SC + verifier
    a_mcts = accuracy_at_compute(compute * 0.85) * 1.02        # MCTS
    print(f"  Pure BoN        (64×): {a_bon:.1%}")
    print(f"  SC + PRM rerank (45×): {a_sc:.1%}")
    print(f"  MCTS+PRM        (54×): {a_mcts:.1%}  ← 通常最优")

    # 拐点
    print("\n=== 收益拐点 (α=0.4 模型) ===")
    for c in (8, 32, 128, 512, 2048):
        a = accuracy_at_compute(c)
        marginal = accuracy_at_compute(c) - accuracy_at_compute(c // 2)
        print(f"  {c:>4}× → {a:.1%}  (边际 +{marginal:.1%})")

    # 实战策略
    print("\n=== 实战建议 ===")
    print("  • 简单任务(<70% 目标): CoT-extension, N=1~4")
    print("  • 中等(70-90%):        BoN N=8~32")
    print("  • 高难度(>90%):        MCTS+PRM, N=128+, 强 verifier")
    print("  • 自适应: 小分类器先估难度, 再分配 compute")


if __name__ == "__main__":
    main()
