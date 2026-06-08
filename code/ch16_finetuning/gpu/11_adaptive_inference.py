# ---
# chapter: 16
# topic: 自适应推理 (Test-Time Compute 复杂度路由)
# section: 16.8.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only)
# run: python 11_adaptive_inference.py
# expected_runtime: <1s
# expected_output: 4 条 query 的 complexity 分数 + 模型档位
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.8.3
#
# Interview hooks:
#   1. Test-Time Compute 的核心范式：训练计算 + 推理计算 = 最终能力？
#   2. Self-Consistency 的投票机制如何实现？多数投票 vs 加权投票？
#   3. 成本-质量权衡：Fast 模式 vs Deep 模式的 token 消耗比？典型 5-10x？
"""自适应推理: 根据 query 复杂度选不同模型/推理档位 (Test-Time Compute).

2026 范式:
  - 简单 query  →  0.5B 模型 + 短 context (fast, 省钱)
  - 中等 query  →  7B 模型 + CoT (balanced)
  - 复杂 query  →  72B 模型 + Self-Consistency × 5 (premium)

节省成本的核心: 不让所有 query 都过最大模型.
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=0, min_count=1)


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


def pick_model(complexity: int) -> dict:
    """根据复杂度选模型档位 + 推理策略."""
    if complexity <= 1:
        return {
            "tier": "fast",
            "model": "Qwen2.5-0.5B-Instruct",
            "strategy": "no-CoT, 1 sample",
            "est_tokens": "≤ 256",
        }
    if complexity <= 4:
        return {
            "tier": "balanced",
            "model": "Qwen2.5-7B-Instruct",
            "strategy": "CoT, 1 sample",
            "est_tokens": "≤ 1024",
        }
    return {
        "tier": "premium",
        "model": "Qwen2.5-72B / deepseek-reasoner",
        "strategy": "CoT + Self-Consistency × 5",
        "est_tokens": "≤ 4096",
    }


def main():
    check_hardware()

    print("=== 自适应推理 (Test-Time Compute 路由) ===\n")
    queries = [
        "hi",
        "What's 2+2?",
        "Explain quantum entanglement in 3 sentences",
        "Design a distributed system for processing 1M QPS with strong consistency",
    ]
    for q in queries:
        score, breakdown = complexity_score(q)
        m = pick_model(score)
        print(f"Q: {q}")
        print(f"  complexity: {score} (breakdown: {breakdown})")
        print(f"  → tier: {m['tier']}")
        print(f"    model:   {m['model']}")
        print(f"    strategy:{m['strategy']}")
        print(f"    tokens:  {m['est_tokens']}\n")


if __name__ == "__main__":
    main()
