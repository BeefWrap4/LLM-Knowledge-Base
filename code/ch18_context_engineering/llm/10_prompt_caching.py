# ---
# chapter: 18
# topic: Context Engineering
# topic_id: context_engineering.prompt_caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 10_prompt_caching.py
# expected_runtime: <1s
# ---
#
# See: ../../../18_Context_Engineering.md
# Official sources (checked 2026-07-31):
#   - Anthropic: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
#   - OpenAI: https://developers.openai.com/api/docs/guides/prompt-caching
#   - Gemini: https://ai.google.dev/gemini-api/docs/caching
#             https://ai.google.dev/gemini-api/docs/generate-content/caching
#
# Important:
#   本文件是纯离线算式，不调用 API，也不冻结某个模型的美元标价。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CachePolicySnapshot:
    provider_scope: str
    checked_on: str
    lifetime: str
    billing_boundary: str
    source_url: str


POLICY_SNAPSHOTS = (
    CachePolicySnapshot(
        provider_scope="Anthropic active Claude models",
        checked_on="2026-07-31",
        lifetime="默认 5m；可选 1h",
        billing_boundary="相对基础输入价：5m 写 1.25×、1h 写 2×、读 0.1×",
        source_url="https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
    ),
    CachePolicySnapshot(
        provider_scope="OpenAI GPT-5.6+",
        checked_on="2026-07-31",
        lifetime="ttl 表示最短生命周期；当前唯一值/默认值为 30m，服务可能保留更久",
        billing_boundary="写入 1.25× 未缓存输入价；读取按目标模型 cached-input 价",
        source_url="https://developers.openai.com/api/docs/guides/prompt-caching",
    ),
    CachePolicySnapshot(
        provider_scope="Gemini 2.5+",
        checked_on="2026-07-31",
        lifetime="implicit 默认；显式缓存默认 TTL 1h",
        billing_boundary="implicit 不保证命中；显式缓存另计 token 存储时长，不是免费命中",
        source_url="https://ai.google.dev/gemini-api/docs/generate-content/caching",
    ),
)


@dataclass(frozen=True)
class NormalizedInputCost:
    turns: int
    cache_hits: int
    baseline_units: float
    cached_units: float

    @property
    def change_vs_no_cache(self) -> float:
        """负数表示下降，正数表示上升；只覆盖输入侧归一化成本。"""

        return self.cached_units / self.baseline_units - 1.0


def anthropic_normalized_input_cost(
    *,
    prefix_tokens: int,
    dynamic_tokens_per_turn: int,
    turns: int,
    cache_hits: int,
    ttl: str = "5m",
) -> NormalizedInputCost:
    """按 Anthropic 当前公开倍率演示输入侧成本。

    ``1 unit`` 等于一个基础输入 token 的相对成本。函数不包含输出、Batch、
    data residency、工具、网络或基础设施费用，因此不能当成总账单预测。
    """

    if prefix_tokens < 0 or dynamic_tokens_per_turn < 0:
        raise ValueError("token 数不能为负数")
    if prefix_tokens + dynamic_tokens_per_turn == 0:
        raise ValueError("每轮至少需要一个输入 token")
    if turns <= 0:
        raise ValueError("turns 必须大于 0")
    if not 0 <= cache_hits <= turns:
        raise ValueError("cache_hits 必须在 [0, turns] 内")
    if ttl not in {"5m", "1h"}:
        raise ValueError("ttl 仅支持 5m 或 1h 教学快照")

    write_factor = 1.25 if ttl == "5m" else 2.0
    read_factor = 0.10
    misses = turns - cache_hits
    baseline = turns * (prefix_tokens + dynamic_tokens_per_turn)
    cached = (
        misses * prefix_tokens * write_factor
        + cache_hits * prefix_tokens * read_factor
        + turns * dynamic_tokens_per_turn
    )
    return NormalizedInputCost(
        turns=turns,
        cache_hits=cache_hits,
        baseline_units=float(baseline),
        cached_units=float(cached),
    )


def best_practices() -> tuple[str, ...]:
    return (
        "把完全稳定的 system、few-shot 与工具 schema 放在前缀，动态内容放在后缀",
        "按提供方要求使用 exact prefix、cache key、breakpoint 与 TTL",
        "记录 cache write/read/miss token；用观测命中率而不是假设命中",
        "把首次写入、未缓存输入、输出、显式存储、工具和基础设施纳入总成本",
        "上线或切换模型前重新核对官方文档与价格页",
    )


def run_demo() -> None:
    print("=== Prompt Caching 官方规则快照（核验日 2026-07-31） ===")
    for policy in POLICY_SNAPSHOTS:
        print(f"- {policy.provider_scope}: {policy.lifetime}; {policy.billing_boundary}")
        print(f"  {policy.source_url}")

    print("\n=== Anthropic 5m 倍率：归一化输入侧示例（不是美元报价/总账单） ===")
    for turns in (1, 3, 10):
        result = anthropic_normalized_input_cost(
            prefix_tokens=8_000,
            dynamic_tokens_per_turn=500,
            turns=turns,
            cache_hits=max(0, turns - 1),
        )
        print(
            f"{turns:>2} 轮, hit={result.cache_hits:>2}: "
            f"baseline={result.baseline_units:.0f}, cached={result.cached_units:.0f}, "
            f"输入侧变化={result.change_vs_no_cache:+.1%}"
        )

    print("\n注意：Anthropic 读取 0.1× 只意味着命中的输入 token 相对基础输入价低 90%。")
    print("它不等于整次请求或系统总成本下降 90%，Gemini cache hit 也不是免费。")

    print("\n=== 最佳实践 ===")
    for index, practice in enumerate(best_practices(), start=1):
        print(f"{index}. {practice}")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
