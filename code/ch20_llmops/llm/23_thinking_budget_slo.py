# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.8 Thinking-Budget SLO
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 23_thinking_budget_slo.py
# expected_runtime: < 1s
# expected_output: Span list with thinking utilization attributes and metrics
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20108-thinking-budget-slo思考预算-slo
# Interview hooks:
#  - 供应商的 Token 预算与 reasoning_effort 等级为何不能视为同一控制量？
#  - "思考超限"（overshoot）和"思考浪费"（waste）分别对应什么业务问题？
#  - 思考预算 SLO 的利用率目标如何用质量、延迟和成本数据校准？

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "thinking-slo"}))
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
metrics.set_meter_provider(MeterProvider())

meter = metrics.get_meter("thinking-budget-slo")
thinking_usage_hist = meter.create_histogram("app.llm.reasoning_tokens", unit="{token}")
thinking_budget_hist = meter.create_histogram("app.llm.reasoning_budget_tokens", unit="{token}")
thinking_overshoot_counter = meter.create_counter("app.llm.reasoning_budget_overshoot", unit="{event}")
thinking_waste_counter = meter.create_counter("app.llm.reasoning_budget_underuse", unit="{event}")
tracer = trace.get_tracer("thinking-slo")


def record_thinking_usage(
    provider_name: str,
    model: str,
    budget_tokens: int,
    used_tokens: int,
    *,
    overshoot_ratio: float,
    underuse_ratio: float,
):
    """记录一次 Token 预算使用情况；阈值由调用方根据业务 SLO 注入。"""
    if budget_tokens <= 0 or used_tokens < 0:
        raise ValueError("budget_tokens must be positive and used_tokens non-negative")
    if not 0 <= underuse_ratio < 1 <= overshoot_ratio:
        raise ValueError("expected 0 <= underuse_ratio < 1 <= overshoot_ratio")
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider_name,
        "gen_ai.request.model": model,
    }
    thinking_usage_hist.record(used_tokens, attributes=attrs)
    thinking_budget_hist.record(budget_tokens, attributes=attrs)
    utilization = used_tokens / budget_tokens
    with tracer.start_as_current_span(f"thinking.{model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", provider_name)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.usage.reasoning.output_tokens", used_tokens)
        span.set_attribute("app.llm.reasoning_budget_tokens", budget_tokens)
        span.set_attribute("app.llm.reasoning_utilization_ratio", round(utilization, 3))
        if utilization > overshoot_ratio:
            thinking_overshoot_counter.add(1, attributes=attrs)
            span.set_attribute("app.llm.reasoning_slo_status", "overshoot")
        elif utilization < underuse_ratio:
            thinking_waste_counter.add(1, attributes=attrs)
            span.set_attribute("app.llm.reasoning_slo_status", "waste")
        else:
            span.set_attribute("app.llm.reasoning_slo_status", "healthy")
    return utilization


if __name__ == "__main__":
    provider_name = os.environ.get("LLM_PROVIDER_NAME", "anthropic")
    balanced_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    premium_model = os.environ.get("ANTHROPIC_PREMIUM_MODEL", "claude-opus-5")
    fast_model = os.environ.get("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5")
    overshoot_ratio = float(os.environ.get("LLM_REASONING_OVERSHOOT_RATIO", "1.0"))
    underuse_ratio = float(os.environ.get("LLM_REASONING_UNDERUSE_RATIO", "0.3"))
    samples = [
        (balanced_model, 2048, 512),
        (balanced_model, 2048, 2100),
        (balanced_model, 2048, 200),
        (premium_model, 8192, 6000),
        (fast_model, 1024, 1100),
    ]
    for model, budget, used in samples:
        util = record_thinking_usage(
            provider_name,
            model,
            budget,
            used,
            overshoot_ratio=overshoot_ratio,
            underuse_ratio=underuse_ratio,
        )
        print(f"  {model}: util={util:.2f}")
    spans = exporter.get_finished_spans()
    statuses = [s.attributes.get("app.llm.reasoning_slo_status") for s in spans]
    print(f"status histogram: {dict((s, statuses.count(s)) for s in set(statuses))}")
    print("OK")
