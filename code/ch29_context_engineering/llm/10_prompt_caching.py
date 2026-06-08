# ---
# chapter: 29
# topic: Prompt Caching — 缓存前缀策略, 节省 90% token 成本
# section: 29.7
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 10_prompt_caching.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.7
# Cross-refs:
#   - Ch20 LLMOps (成本监控)
#   - Ch25 推理引擎 (Prefix Cache / KV Cache)
#   - Ch02 Token 经济学
#
# Interview hooks:
#   - "Prompt Caching 缓存什么?"      →  稳定的 system prompt + few-shot + 工具 schema
#   - "Anthropic 缓存折扣?"            →  写入 ×1.25, 读取 ×0.1 (省 90%)
#   - "多轮对话缓存如何累积?"          →  prefix 随对话单调增长, 命中可叠加

from __future__ import annotations

from dataclasses import dataclass

# Anthropic 公开定价 (USD per 1M tokens, mock)
ANTHROPIC = {
    "base_in": 3.0,
    "base_out": 15.0,
    "cache_write_factor": 1.25,  # 写时多收 25%
    "cache_read_factor": 0.10,  # 读时只收 10%
    "min_cacheable": 1024,  # 最小可缓存 token (官方约束)
    "cache_ttl_s": 300,  # 5 分钟
}


@dataclass
class CacheStats:
    total_in: int
    cached_in: int
    fresh_in: int
    cache_hits: int
    cost_with_cache: float
    cost_no_cache: float
    saving_pct: float


def calc_cost(prefix_tokens: int, dynamic_tokens: int, output_tokens: int, n_turns: int = 1) -> CacheStats:
    """模拟多轮对话的 prefix 缓存累积。
    - 第 1 轮: prefix 全部新鲜, 走 cache_write × 1.25
    - 第 N 轮 (N>1): 相同 prefix 走 cache_read × 0.1
    - dynamic 部分每轮都新鲜
    """
    a = ANTHROPIC
    cost_with = 0.0
    cost_without = 0.0
    hits = 0
    fresh_total = 0
    cached_total = 0
    for t in range(1, n_turns + 1):
        # 无缓存基线
        cost_without += (prefix_tokens + dynamic_tokens) / 1e6 * a["base_in"]
        # 有缓存
        if t == 1:
            cost_with += prefix_tokens / 1e6 * a["base_in"] * a["cache_write_factor"]
            cost_with += dynamic_tokens / 1e6 * a["base_in"]
        else:
            # prefix 全部命中
            cost_with += prefix_tokens / 1e6 * a["base_in"] * a["cache_read_factor"]
            cost_with += dynamic_tokens / 1e6 * a["base_in"]
            hits += 1
        cost_with += output_tokens / 1e6 * a["base_out"]
        cost_without += output_tokens / 1e6 * a["base_out"]
        fresh_total += dynamic_tokens
        cached_total += prefix_tokens if t > 1 else 0
    return CacheStats(
        total_in=(prefix_tokens + dynamic_tokens) * n_turns,
        cached_in=cached_total,
        fresh_in=fresh_total + prefix_tokens,  # 首次的 prefix 也算 fresh
        cache_hits=hits,
        cost_with_cache=cost_with,
        cost_no_cache=cost_without,
        saving_pct=1 - cost_with / cost_without if cost_without else 0,
    )


def best_practices() -> list[str]:
    return [
        "1. 把稳定的 system prompt + few-shot examples 放在 prefix (缓存命中区)",
        "2. 动态内容 (RAG 检索/用户 query) 放在 prefix 之后 (每轮新鲜)",
        "3. 利用多轮对话 prefix 单调增长, 累积缓存命中",
        "4. Anthropic: prefix ≥ 1024 tokens 才值得缓存",
        "5. OpenAI: 自动缓存, 同样 prefix 命中",
        "6. Gemini: 显式 cacheContents, TTL 1 小时, 命中免费",
        "7. 长 system prompt 场景, 缓存收益最大 (e.g. 20k tokens 的 code base 文档)",
    ]


def run_demo() -> None:
    print("=== Prompt Caching 成本对比 (Anthropic Claude Sonnet 4) ===\n")
    print(f"场景: prefix={8000} tokens (system+docs), dynamic={500} tokens/turn, output={300} tokens/turn\n")
    for n_turns in [1, 3, 10, 50]:
        s = calc_cost(prefix_tokens=8000, dynamic_tokens=500, output_tokens=300, n_turns=n_turns)
        print(f"--- {n_turns} 轮对话 ---")
        print(f"  无缓存: ${s.cost_no_cache:.4f}")
        print(f"  有缓存: ${s.cost_with_cache:.4f}  (节省 {s.saving_pct:.0%}, 命中 {s.cache_hits} 次)")

    print("\n=== 最佳实践 ===")
    for p in best_practices():
        print("  " + p)


if __name__ == "__main__":
    run_demo()
    print("\nOK")
