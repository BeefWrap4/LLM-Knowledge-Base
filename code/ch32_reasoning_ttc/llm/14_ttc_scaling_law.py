# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.ttc_scaling_law
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 14_ttc_scaling_law.py
# expected_runtime: <1s
# expected_output: 打印 accuracy vs compute 的 scaling 曲线
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. TTC scaling law 的数学形式？和预训练 scaling law 区别？
#   2. 何时 BoN > CoT-extension > MCTS-PRM？
#   3. verifier 饱和点 vs 模型能力饱和点？
"""Test-Time Compute Scaling 教学模拟。

下面的饱和曲线和参数是人为设定的可视化示例，不是 Snell et al. (2024) 拟合出的
通用定律。论文强调方法效果随题目难度、基础模型与 verifier 变化。
Source: https://arxiv.org/abs/2408.03314
"""

from __future__ import annotations


def accuracy_at_compute(compute: float, base_acc: float = 0.5, alpha: float = 0.4, c0: float = 1.0) -> float:
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
        "MCTS+PRM": 128,
    }

    print("=== Test-Time Compute Scaling（合成曲线，不是 benchmark）===\n")
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
    a_bon = accuracy_at_compute(compute)  # 全 BoN
    a_sc = accuracy_at_compute(compute * 0.7) * 0.97  # SC + verifier
    a_mcts = accuracy_at_compute(compute * 0.85) * 1.02  # MCTS
    print(f"  Pure BoN        (64×): {a_bon:.1%}")
    print(f"  SC + PRM rerank (45×): {a_sc:.1%}")
    print(f"  MCTS+PRM        (54×): {a_mcts:.1%}  ← 仅为人为参数下的结果")

    # 拐点
    print("\n=== 收益拐点 (α=0.4 模型) ===")
    for c in (8, 32, 128, 512, 2048):
        a = accuracy_at_compute(c)
        marginal = accuracy_at_compute(c) - accuracy_at_compute(c // 2)
        print(f"  {c:>4}× → {a:.1%}  (边际 +{marginal:.1%})")

    # 实战策略
    print("\n=== 实战建议 ===")
    print("  • 在目标集上按难度分桶，同时记录质量、成本、延迟与方差")
    print("  • 固定 FLOPs 比较 CoT、BoN、搜索与自适应策略")
    print("  • 先验证 verifier 校准，再扩大采样或搜索预算")
    print("  • 线上按 SLO 分配预算，并保留超时/取消与回退")
    print("OK")


if __name__ == "__main__":
    main()
