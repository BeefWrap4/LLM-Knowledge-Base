# ---
# chapter: 29
# topic: Token 经济学 — 用可注入费率表核算上下文成本
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
#   - "Context 长度如何影响成本?" → 分开核算未缓存输入、缓存写/读、输出和存储
#   - "缓存折扣等于总成本折扣吗?" → 不等于；还要计入 miss、write、output、storage 等
#   - "长 context 是否值得用?"   → 用目标任务质量、延迟与实际账单共同判断

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateCard:
    """由调用方在部署日从官方价格页填入的费率表。

    所有费率均为 ``currency / 1M tokens``；storage 是
    ``currency / (1M token × hour)``。不同提供方若没有某一项，可填 0。
    """

    label: str
    currency: str
    checked_on: str
    source_url: str
    uncached_input: float
    cache_read: float
    cache_write: float
    output: float
    storage_per_million_token_hour: float = 0.0

    def __post_init__(self) -> None:
        rates = (
            self.uncached_input,
            self.cache_read,
            self.cache_write,
            self.output,
            self.storage_per_million_token_hour,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("费率不能为负数")


@dataclass(frozen=True)
class TokenUsage:
    """一次或一批请求的实际 usage 分类。"""

    uncached_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    stored_token_hours: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.uncached_input_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.output_tokens,
            self.stored_token_hours,
        )
        if any(value < 0 for value in values):
            raise ValueError("usage 不能为负数")


@dataclass(frozen=True)
class CostBreakdown:
    uncached_input: float
    cache_read: float
    cache_write: float
    output: float
    storage: float

    @property
    def total(self) -> float:
        return self.uncached_input + self.cache_read + self.cache_write + self.output + self.storage


def estimate_cost(rate_card: RateCard, usage: TokenUsage) -> CostBreakdown:
    """按显式 usage 与费率表计算成本，不估算质量或延迟。"""

    per_million = 1_000_000
    return CostBreakdown(
        uncached_input=usage.uncached_input_tokens / per_million * rate_card.uncached_input,
        cache_read=usage.cache_read_tokens / per_million * rate_card.cache_read,
        cache_write=usage.cache_write_tokens / per_million * rate_card.cache_write,
        output=usage.output_tokens / per_million * rate_card.output,
        storage=usage.stored_token_hours / per_million * rate_card.storage_per_million_token_hour,
    )


def run_demo() -> None:
    # 这些是任意的归一化教学参数，不是任何厂商或模型的美元报价。
    demo_rates = RateCard(
        label="synthetic-normalized-rates-not-live-pricing",
        currency="cost-unit",
        checked_on="N/A",
        source_url="replace-with-provider-official-pricing-page",
        uncached_input=1.0,
        cache_read=0.35,
        cache_write=1.20,
        output=4.0,
        storage_per_million_token_hour=0.02,
    )
    scenarios = {
        "无缓存": TokenUsage(
            uncached_input_tokens=80_000,
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=2_000,
        ),
        "有写入和命中": TokenUsage(
            uncached_input_tokens=16_000,
            cache_read_tokens=64_000,
            cache_write_tokens=8_000,
            output_tokens=2_000,
            stored_token_hours=8_000,
        ),
    }

    print("=== Token 成本公式演示（归一化教学参数，不是厂商现价） ===")
    print("上线时必须用目标模型、区域和服务层的官方价格覆盖 RateCard。\n")
    for name, usage in scenarios.items():
        cost = estimate_cost(demo_rates, usage)
        print(f"[{name}]")
        print(
            f"  uncached={cost.uncached_input:.4f}, read={cost.cache_read:.4f}, "
            f"write={cost.cache_write:.4f}, output={cost.output:.4f}, storage={cost.storage:.4f}"
        )
        print(f"  total={cost.total:.4f} {demo_rates.currency}\n")

    print("质量和端到端延迟没有跨模型通用公式；请从真实评测与 usage/trace 读取。")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
