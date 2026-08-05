# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: lora_qlora.adaptive_inference
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: (stdlib only)
# run: python 11_adaptive_inference.py
# expected_runtime: <1s
# expected_output: 4 条 query 的 complexity 分数 + 模型档位
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
#
# Interview hooks:
#   1. 为什么更高推理预算不保证更高任务收益？如何做增量评测？
#   2. Self-Consistency 的投票机制如何实现？多数投票 vs 加权投票？
#   3. 如何用任务集实测 Fast 与 Thinking 路径的质量、延迟和 token 成本？
"""自适应推理路由的纯逻辑示例。

这里输出的是待评测的策略档位，不绑定某个易漂移的产品名，也不声称关键词启发式能直接
用于生产。上线前必须用代表性任务集校准，并设置质量、安全、延迟和成本回退门槛。
"""


COMPLEX_KEYWORDS = {
    "expl": ["explain", "analyze", "compare", "解释", "分析", "对比", "为什么", "how"],
    "create": ["design", "implement", "build", "设计", "实现", "构建", "写一个"],
    "reason": ["prove", "derive", "证明", "推导", "求解", "evaluate"],
}


def complexity_score(query: str) -> tuple[int, dict]:
    """简单启发式: 字数 + 标点 + 关键词."""
    breakdown = {
        "length_pts": len(query) // 20,
        "punct_pts": sum(1 for c in query if c in "?!.;:?"),
    }
    score = breakdown["length_pts"] + breakdown["punct_pts"]
    for kws in COMPLEX_KEYWORDS.values():
        for kw in kws:
            if kw in query.lower():
                score += 1
                breakdown.setdefault("kw_pts", 0)
                breakdown["kw_pts"] += 1
                break  # 每个类别只算 1 次
    return score, breakdown


def pick_route(complexity: int) -> dict:
    """返回候选策略；阈值与预算必须用业务评测校准."""
    if complexity <= 1:
        return {
            "tier": "baseline",
            "model_role": "low-latency model or low reasoning effort",
            "strategy": "single sample",
            "budget_policy": "latency-first baseline",
        }
    if complexity <= 4:
        return {
            "tier": "balanced",
            "model_role": "balanced model/effort",
            "strategy": "single sample + tool verification when available",
            "budget_policy": "quality/latency trade-off",
        }
    return {
        "tier": "quality-first",
        "model_role": "high-capability model/effort",
        "strategy": "optional multi-sample or independent verifier",
        "budget_policy": "enable only after measured net gain",
    }


def main():
    print("=== 自适应推理路由（纯策略演示）===\n")
    queries = [
        "hi",
        "What's 2+2?",
        "Explain quantum entanglement in 3 sentences",
        "Design a distributed system for processing 1M QPS with strong consistency",
    ]
    for q in queries:
        score, breakdown = complexity_score(q)
        route = pick_route(score)
        print(f"Q: {q}")
        print(f"  complexity: {score} (breakdown: {breakdown})")
        print(f"  → tier: {route['tier']}")
        print(f"    model role: {route['model_role']}")
        print(f"    strategy:   {route['strategy']}")
        print(f"    budget:     {route['budget_policy']}\n")
    print("OK")


if __name__ == "__main__":
    main()
