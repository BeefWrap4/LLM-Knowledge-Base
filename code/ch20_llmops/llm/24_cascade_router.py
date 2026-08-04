# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.8.9 Cascade / Router 模型成本模式
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 24_cascade_router.py
# expected_runtime: < 1s
# expected_output: Routing decisions and span attributes printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20109-cascade--router-模型成本模式
# Interview hooks:
#  - Cascade Router 的节省率为什么只能用自己的流量分布与账单验证？
#  - 升级率、质量 Guardrail 和加权成本如何联合设计 SLO？
#  - 路由决策器的训练数据怎么准备并防止分布漂移？

import os
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "cascade-router"}))
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("cascade-router")


@dataclass(frozen=True)
class RouteConfig:
    fast_model: str
    balanced_model: str
    premium_model: str
    fast_max_complexity: float = 0.3
    balanced_max_complexity: float = 0.7

    def __post_init__(self):
        if not 0 <= self.fast_max_complexity < self.balanced_max_complexity <= 1:
            raise ValueError(
                "expected 0 <= fast_max_complexity < balanced_max_complexity <= 1"
            )
        if not all((self.fast_model, self.balanced_model, self.premium_model)):
            raise ValueError("all route model ids must be non-empty")


def cascade_route(query: str, complexity: float, config: RouteConfig) -> tuple[str, str]:
    """按注入的路由配置选模型；阈值需用离线集和线上 Guardrail 校准。"""
    del query  # 默认不把原始内容写入遥测。
    if not 0 <= complexity <= 1:
        raise ValueError("complexity must be in [0, 1]")
    span = trace.get_current_span()
    span.set_attribute("app.llm.router.query_complexity", complexity)
    span.set_attribute("app.llm.router.config_version", "demo-v1")

    if complexity < config.fast_max_complexity:
        tier, model = "tier_1_fast", config.fast_model
    elif complexity < config.balanced_max_complexity:
        tier, model = "tier_2_balanced", config.balanced_model
    else:
        tier, model = "tier_3_premium", config.premium_model
    span.set_attribute("app.llm.router.tier", tier)
    span.set_attribute("gen_ai.request.model", model)
    return tier, model


def compute_input_cost_delta(
    *,
    input_tokens: int,
    from_rate_usd_per_million: float,
    to_rate_usd_per_million: float,
) -> float:
    """用调用方注入的 Rate Card 计算本次输入侧估算差额。"""
    if input_tokens < 0:
        raise ValueError("input_tokens must be non-negative")
    return input_tokens / 1_000_000 * (to_rate_usd_per_million - from_rate_usd_per_million)


def record_upgrade(
    original_tier: str,
    upgraded_tier: str,
    reason: str,
    estimated_cost_delta_usd: float | None = None,
):
    attributes: dict[str, str | float] = {
        "app.llm.router.from_tier": original_tier,
        "app.llm.router.to_tier": upgraded_tier,
        "app.llm.router.upgrade_reason": reason,
    }
    if estimated_cost_delta_usd is not None:
        attributes["app.llm.router.estimated_cost_delta_usd"] = estimated_cost_delta_usd
    trace.get_current_span().add_event("cascade.upgrade", attributes=attributes)


if __name__ == "__main__":
    config = RouteConfig(
        fast_model=os.environ.get("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5"),
        balanced_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
        premium_model=os.environ.get("ANTHROPIC_PREMIUM_MODEL", "claude-opus-5"),
    )
    # 教学 Rate Card 可通过环境变量替换；默认值不代表供应商当前价格。
    balanced_rate = float(os.environ.get("LLM_BALANCED_INPUT_USD_PER_MILLION", "2"))
    premium_rate = float(os.environ.get("LLM_PREMIUM_INPUT_USD_PER_MILLION", "4"))
    queries = [
        ("简单翻译", 0.1),
        ("总结文档", 0.4),
        ("复杂推理", 0.85),
        ("代码生成", 0.6),
        ("闲聊", 0.05),
    ]
    counts: dict[str, int] = {}
    for query, complexity in queries:
        with tracer.start_as_current_span("route") as span:
            tier, model = cascade_route(query, complexity, config)
            counts[tier] = counts.get(tier, 0) + 1
            if tier == "tier_3_premium":
                delta = compute_input_cost_delta(
                    input_tokens=1000,
                    from_rate_usd_per_million=balanced_rate,
                    to_rate_usd_per_million=premium_rate,
                )
                record_upgrade("tier_2_balanced", tier, "low_confidence", delta)
            print(f"  complexity={complexity} -> {model}")

    print(f"traffic distribution: {counts}")
    upgrades = [
        event
        for finished_span in exporter.get_finished_spans()
        for event in finished_span.events
        if event.name == "cascade.upgrade"
    ]
    print(f"upgrades: {len(upgrades)}")
    print("OK")
