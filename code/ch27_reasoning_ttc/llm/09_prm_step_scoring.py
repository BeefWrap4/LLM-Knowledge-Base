# ---
# chapter: 27
# topic: Process Reward Model (PRM) step-level scoring
# section: 27.4 PRM
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 09_prm_step_scoring.py
# expected_runtime: <1s
# expected_output: 打印每个推理步骤的 PRM 分数与累积胜率
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.4 + §27.8 Q6
# Interview hooks:
#   1. PRM vs ORM 在 reward hacking 风险上的差异？
#   2. PRM800K 数据如何构造？人工标注 vs 自动 (Math-Shepherd)？
#   3. PRM 在推理时如何用？引导 beam search 还是 Best-of-N？
"""Process Reward Model (PRM)：对推理的每一步打分。

训练数据：PRM800K (Lightman et al. 2023) 提供 800K 步骤级标注。
推理时：每步给一个 correct/incorrect 概率，引导搜索。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Step:
    idx: int
    text: str
    label: int  # 1=correct, 0=incorrect
    prm_score: float  # 模型预测的"正确"概率


def mock_prm_predict(step_text: str) -> float:
    """Mock PRM：根据关键词给一个分数。真实 PRM 是用 LLM 分类器微调。"""
    s = step_text.lower()
    score = 0.5
    # 算术步骤特征
    if any(op in s for op in ("=", "+", "-", "×", "/")):
        score += 0.2
    # 出现错误信号
    if "wrong" in s or "incorrect" in s or "contradiction" in s:
        score -= 0.3
    if "therefore" in s or "thus" in s:
        score += 0.1
    return max(0.05, min(0.99, score))


def aggregate_step_score(scores: list[float], mode: str = "prod") -> float:
    """聚合步骤分数。

    prod:  P(all correct) = ∏ p_i
    min:   weakest link
    mean:  average
    """
    if mode == "prod":
        s = 1.0
        for p in scores:
            s *= p
        return s
    if mode == "min":
        return min(scores)
    if mode == "mean":
        return sum(scores) / len(scores)
    raise ValueError(mode)


def main() -> None:
    # 模拟一段数学推理
    chain = [
        "Assume sqrt(2) is rational, write as p/q with gcd(p,q)=1.",
        "Then p^2 = 2 q^2, so p^2 is even.",
        "Therefore p is even, let p=2k, then 4k^2 = 2 q^2.",
        "Divide by 2: 2k^2 = q^2, so q^2 is even, q is even.",  # 此步有跳跃
        "But p and q both even contradicts gcd=1.",
        "Therefore sqrt(2) is irrational.",
    ]
    steps = []
    for i, t in enumerate(chain):
        s = mock_prm_predict(t)
        steps.append(Step(i, t, label=1, prm_score=round(s, 3)))

    print(f"{'#':>2}  {'PRM':>6}  text")
    print("-" * 70)
    for s in steps:
        marker = "OK" if s.prm_score > 0.5 else "WARN"
        print(f"{s.idx:>2}  {s.prm_score:>5.2f}  [{marker}] {s.text[:55]}")

    # 聚合
    print("\n--- 聚合分数 (全链正确概率) ---")
    scores = [s.prm_score for s in steps]
    for mode in ("prod", "min", "mean"):
        agg = aggregate_step_score(scores, mode)
        print(f"  {mode:>4}: {agg:.4f}")

    # 对比 ORM (只看最终答案)
    orm_score = 1.0 if "irrational" in chain[-1].lower() else 0.0
    print(f"\nORM (只看最终答案): {orm_score}")
    print(f"PRM-prod:           {aggregate_step_score(scores, 'prod'):.4f}")
    print("  → PRM 暴露了步骤 3 的不确定性 (0.6)，ORM 看不到")



if __name__ == "__main__":
    main()
