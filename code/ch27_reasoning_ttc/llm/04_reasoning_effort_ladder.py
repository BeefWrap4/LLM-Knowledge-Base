# ---
# chapter: 27
# topic: Reasoning Effort 决策阶梯 + 成本/延迟预算
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 04_reasoning_effort_ladder.py
# expected_runtime: <1s
# expected_output: 打印 ladder + 路由决策表
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2 + §27.7
# Interview hooks:
#   1. 如何为不同任务选择 reasoning_effort？成本如何估算？
#   2. 推理模型路由 (model router) 的设计要点？
#   3. reasoning_effort="high" 一定能提升准确率吗？何时会下降？
"""Reasoning Effort Ladder：根据任务难度/预算/延迟约束选择档位。

工程实践:
  • 用小分类器判定 query 难度 (cheap)
  • 简单 → low，复杂 → high，中间 → medium
  • 监控 reasoning_tokens 实际值，动态调整
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Effort(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class EffortProfile:
    name: str
    thought_tokens: tuple[int, int]
    accuracy: float
    latency_s: tuple[float, float]
    cost_per_query: tuple[float, float]  # USD


PROFILES = {
    Effort.LOW: EffortProfile(
        "low", (100, 500), 0.55, (0.5, 2.0), (0.001, 0.005)
    ),
    Effort.MEDIUM: EffortProfile(
        "medium", (1_000, 5_000), 0.80, (3.0, 15.0), (0.01, 0.05)
    ),
    Effort.HIGH: EffortProfile(
        "high", (10_000, 50_000), 0.95, (30.0, 180.0), (0.10, 0.50)
    ),
}


def classify_difficulty(query: str) -> Effort:
    """极简分类器：按关键词粗判。真实系统用小模型微调。"""
    q = query.lower()
    math_kw = ("prove", "integral", "derivative", "equation", "theorem")
    code_kw = ("implement", "algorithm", "complexity", "optimize", "leetcode")
    logic_kw = ("why", "analyze", "compare", "evaluate", "design")

    if any(k in q for k in math_kw) or any(k in q for k in code_kw):
        return Effort.HIGH
    if any(k in q for k in logic_kw) or len(q.split()) > 40:
        return Effort.MEDIUM
    return Effort.LOW


def estimate_cost(effort: Effort, n_queries: int) -> float:
    """估算月成本。取每档中位数 × 月调用量。"""
    p = PROFILES[effort]
    cost = (p.cost_per_query[0] + p.cost_per_query[1]) / 2
    return cost * n_queries


def recommend(
    query: str,
    budget_usd: float | None = None,
    max_latency_s: float | None = None,
) -> Effort:
    """根据 query + 预算/延迟约束选 effort。"""
    eff = classify_difficulty(query)
    if budget_usd is not None:
        cost = estimate_cost(eff, 1)
        while cost > budget_usd and eff > Effort.LOW:
            eff = Effort(eff - 1)
            cost = estimate_cost(eff, 1)
    if max_latency_s is not None:
        p = PROFILES[eff]
        while p.latency_s[1] > max_latency_s and eff > Effort.LOW:
            eff = Effort(eff - 1)
            p = PROFILES[eff]
    return eff


def main() -> None:
    # 阶梯表
    print("Reasoning Effort Ladder")
    print("=" * 78)
    print(f"{'level':<8}{'thought_tok':<14}{'accuracy':<10}"
          f"{'latency_s':<14}{'cost/query':<12}")
    for e, p in PROFILES.items():
        print(
            f"{p.name:<8}"
            f"{str(p.thought_tokens):<14}"
            f"{p.accuracy:<10}"
            f"{str(p.latency_s):<14}"
            f"{str(p.cost_per_query):<12}"
        )

    # 路由示例
    queries = [
        "What is the capital of France?",
        "Compare microservices vs monolithic architecture for a fintech app.",
        "Prove the fundamental theorem of algebra.",
        "Implement a red-black tree insertion in Python.",
    ]
    print("\n路由决策（预算=$0.05, 延迟<10s）:")
    for q in queries:
        e = recommend(q, budget_usd=0.05, max_latency_s=10.0)
        print(f"  [{e.name:>6}] {q[:60]}")

    # 成本对比
    print("\n10K queries 月成本估算:")
    for e in Effort:
        print(f"  {e.name:<7} ${estimate_cost(e, 10_000):.2f}")



if __name__ == "__main__":
    main()
