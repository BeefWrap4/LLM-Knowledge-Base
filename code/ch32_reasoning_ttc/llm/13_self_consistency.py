# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.self_consistency
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 13_self_consistency.py
# expected_runtime: <1s
# expected_output: 不同 N 下一致性投票准确率
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. Self-Consistency 为何能提升推理准确率？前提是什么？
#   2. 与 Best-of-N 区别？何时用哪个？
#   3. 多样性来源：temperature / top_p / prompt ensemble？
"""Self-Consistency (Wang et al. 2022) — 不需要 verifier 的多数投票。

核心: 同问题采样 N 个不同 CoT，统计答案频次，选众数。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass
class SCResult:
    n: int
    answer_counts: dict
    final: str
    confidence: float
    accuracy: float


def mock_sample_chain(question: str, n: int) -> list[str]:
    """Mock: 模型对同一问题生成 N 个推理路径。"""
    answers = ["42", "42", "42", "41", "40", "42"]
    return [random.choice(answers) for _ in range(n)]


def self_consistency(question: str, n: int = 16, ground_truth: str = "42") -> SCResult:
    samples = mock_sample_chain(question, n)
    cnt = Counter(samples)
    final, top_count = cnt.most_common(1)[0]
    return SCResult(
        n=n,
        answer_counts=dict(cnt),
        final=final,
        confidence=top_count / n,
        accuracy=1.0 if final == ground_truth else 0.0,
    )


def main() -> None:
    random.seed(7)

    print("=== Self-Consistency 不同 N ===")
    print(f"  {'N':>4} | {'final':>6} | {'confidence':>10} | {'acc':>6}")
    print("  " + "-" * 38)
    for N in (1, 4, 8, 16, 32, 64):
        r = self_consistency("What is 6*7?", n=N, ground_truth="42")
        dist = " ".join(f"{k}:{v}" for k, v in sorted(r.answer_counts.items()))
        print(f"  {N:>4} | {r.final:>6} | {r.confidence:>9.1%}  | {r.accuracy:>5.1%}   [{dist}]")

    # 多样性来源
    print("\n=== 多样性策略 ===")
    print("  1. temperature  ↑ → 更多样 (但质量下降)")
    print("  2. top_p / top_k  ↑ → 长尾")
    print("  3. prompt ensemble → 改写问题多次采样")
    print("  4. reasoning_effort=high + 多次采样 → SC 收益最大")

    # 局限
    print("\n=== 局限 ===")
    print("  • 答案空间连续时退化 (浮点/代码)")
    print("  • 模型系统性偏差 → SC 也偏向偏差")
    print("  • 长 CoT 重复采样成本 = N × 单次成本")

    # 与 BoN 关系
    print("\n=== SC + PRM/BoN 组合示例（参数需按评估集调优）===")
    print("  step 1: 多次采样并按答案聚类")
    print("  step 2: 对高频候选使用 PRM/verifier 再排序")
    print("  是否优于单独 SC 或 BoN，必须在同等计算预算下实验验证")
    print("OK")


if __name__ == "__main__":
    main()
