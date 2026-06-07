# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.9 Cascade / Router 模型成本模式
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 24_cascade_router.py
# expected_runtime: < 1s
# expected_output: Routing decisions and span attributes printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20109-cascade--router-模型成本模式
# Interview hooks:
#  - Cascade Router 相比单一模型能节省多少成本？典型比例是多少？
#  - 升级率（upgrade rate）和"加权成本"在 SLO 设计里哪个更重要？
#  - 路由决策器（轻量分类器）的训练数据怎么准备？

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


def cascade_route(query: str, complexity: float) -> str:
    """三档 Cascade Router：根据 complexity 决定使用哪一档模型。"""
    span = trace.get_current_span()
    span.set_attribute("gen_ai.router.query_complexity", complexity)

    if complexity < 0.3:
        span.set_attribute("gen_ai.router.tier", "tier_1_cheap")
        span.set_attribute("gen_ai.router.cost_per_1m_input", 0.25)
        return "claude-haiku-4-5"
    elif complexity < 0.7:
        span.set_attribute("gen_ai.router.tier", "tier_2_mid")
        span.set_attribute("gen_ai.router.cost_per_1m_input", 3.0)
        return "claude-sonnet-4-6"
    else:
        span.set_attribute("gen_ai.router.tier", "tier_3_premium")
        span.set_attribute("gen_ai.router.cost_per_1m_input", 15.0)
        return "claude-opus-4-6"


def _price_for(tier: str) -> float:
    return {"tier_1_cheap": 0.25, "tier_2_mid": 3.0, "tier_3_premium": 15.0}.get(tier, 0.0)


def compute_cost_delta(from_tier: str, to_tier: str) -> float:
    return _price_for(to_tier) - _price_for(from_tier)


def record_upgrade(original_tier: str, upgraded_tier: str, reason: str):
    """记录级联升级事件（cost delta 自动计算）"""
    span = trace.get_current_span()
    span.add_event(
        "cascade.upgrade",
        attributes={
            "gen_ai.router.from_tier": original_tier,
            "gen_ai.router.to_tier": upgraded_tier,
            "gen_ai.router.upgrade_reason": reason,
            "gen_ai.router.cost_delta_usd": compute_cost_delta(original_tier, upgraded_tier),
        },
    )


if __name__ == "__main__":
    queries = [
        ("简单翻译", 0.1),
        ("总结文档", 0.4),
        ("复杂推理", 0.85),
        ("代码生成", 0.6),
        ("闲聊", 0.05),
    ]
    counts: dict = {}
    for q, c in queries:
        with tracer.start_as_current_span(f"route.{q}") as sp:
            tier_attr = "tier_1_cheap" if c < 0.3 else ("tier_2_mid" if c < 0.7 else "tier_3_premium")
            counts[tier_attr] = counts.get(tier_attr, 0) + 1
            model = cascade_route(q, c)
            if c >= 0.7:
                # 模拟：从 mid 升级到 premium
                record_upgrade("tier_2_mid", "tier_3_premium", "low_confidence")
            print(f"  q={q!r} c={c} -> {model}")

    total = sum(counts.values())
    print(f"\ntraffic distribution: {counts}  total={total}")
    spans = exporter.get_finished_spans()
    upgrades = [e for s in spans for e in s.events if e.name == "cascade.upgrade"]
    print(f"upgrades: {len(upgrades)}")
