# ---
# chapter: 29
# topic: Token 经济学 — 成本/延迟与 Context 长度的关系
# section: 29.3
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 02_token_economics.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.3
# Cross-refs:
#   - Ch20 LLMOps (成本监控)
#   - Ch25 推理引擎 (KV Cache 复用)
#   - Ch27 Test-Time Compute (延迟预算)
#
# Interview hooks:
#   - "Context 长度如何影响成本?"   →  输入按 token 线性计价, 缓存可省 90%
#   - "TTFT vs TPOT 是什么?"        →  TTFT = Prefill 时间; TPOT = 逐 token 生成时间
#   - "200K context 是否值得用?"     →  看质量, 64K 后存在 Context Rot

from __future__ import annotations

from dataclasses import dataclass

# 2026 主流模型定价 (USD per 1M tokens, 公开口径, mock)
PRICING = {
    "claude-sonnet-4": {
        "in": 3.0,
        "out": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.30,
        "ctx": 200_000,
    },
    "gpt-5": {"in": 2.5, "out": 10.0, "cache_write": 0.0, "cache_read": 0.0, "ctx": 1_000_000},
    "gemini-2.5-pro": {
        "in": 1.25,
        "out": 5.0,
        "cache_write": 0.0,
        "cache_read": 0.0,
        "ctx": 1_000_000,
    },
    "deepseek-v3.2": {
        "in": 0.27,
        "out": 1.1,
        "cache_write": 0.0,
        "cache_read": 0.0,
        "ctx": 128_000,
    },
}


# 经验值 (mock): 在不同 context 长度下, 模型对中后段信息的"关注度"
def attention_quality(context_len_tokens: int) -> float:
    """返回 0-1 之间的质量分数, 模拟 Context Rot 现象。"""
    if context_len_tokens <= 8_000:
        return 1.0
    if context_len_tokens <= 32_000:
        return 1.0 - 0.05 * (context_len_tokens - 8_000) / 24_000
    if context_len_tokens <= 64_000:
        return 0.95 - 0.10 * (context_len_tokens - 32_000) / 32_000
    if context_len_tokens <= 200_000:
        return 0.85 - 0.20 * (context_len_tokens - 64_000) / 136_000
    return max(0.45, 0.65 - 0.15 * (context_len_tokens - 200_000) / 800_000)


@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    cache_hit_ratio: float  # 0-1
    input_cost: float
    output_cost: float
    cache_saving: float
    total_usd: float
    latency_s: float
    quality: float

    def report(self) -> str:
        return (
            f"[{self.model}]\n"
            f"  in={self.input_tokens:,} out={self.output_tokens:,} cache_hit={self.cache_hit_ratio:.0%}\n"
            f"  cost: input=${self.input_cost:.4f} output=${self.output_cost:.4f} "
            f"cache_saving=${self.cache_saving:.4f} -> total=${self.total_usd:.4f}\n"
            f"  latency ≈ {self.latency_s:.2f}s   quality ≈ {self.quality:.2f}"
        )


def estimate(model: str, input_tokens: int, output_tokens: int, cache_hit: float = 0.0) -> CostEstimate:
    p = PRICING[model]
    # 缓存命中部分走 cache_read 价, 未命中部分走 in 价
    cached = int(input_tokens * cache_hit)
    fresh = input_tokens - cached
    in_cost = fresh / 1e6 * p["in"]
    cache_saving = cached / 1e6 * (p["in"] - p["cache_read"])
    out_cost = output_tokens / 1e6 * p["out"]
    # 延迟: TTFT ∝ √(input_tokens) (粗略), TPOT 固定 25ms
    ttft = 0.05 + 0.0001 * (input_tokens**0.5)
    tpot = 0.025
    latency = ttft + tpot * output_tokens
    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit_ratio=cache_hit,
        input_cost=in_cost,
        output_cost=out_cost,
        cache_saving=cache_saving,
        total_usd=in_cost + out_cost,
        latency_s=latency,
        quality=attention_quality(input_tokens),
    )


def run_demo() -> None:
    print("=== 不同 Context 长度下的成本与质量 (mock) ===\n")
    for n_in in [2_000, 16_000, 64_000, 128_000, 200_000, 500_000]:
        for model in ["claude-sonnet-4", "gpt-5", "deepseek-v3.2"]:
            est = estimate(model, n_in, 500)
            print(est.report())
        print()


if __name__ == "__main__":
    run_demo()
