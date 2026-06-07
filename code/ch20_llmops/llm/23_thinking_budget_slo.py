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
#  - Claude thinking.budget_tokens 与 OpenAI reasoning_effort 的关系？
#  - "思考超限"（overshoot）和"思考浪费"（waste）分别对应什么业务问题？
#  - 思考预算 SLO 的利用率目标（0.5~0.8）是怎么定出来的？

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
thinking_usage_hist = meter.create_histogram("gen_ai.thinking.tokens_used", unit="tokens")
thinking_budget_hist = meter.create_histogram("gen_ai.thinking.budget_tokens", unit="tokens")
thinking_overshoot_counter = meter.create_counter("gen_ai.thinking.overshoot", unit="count")
thinking_waste_counter = meter.create_counter("gen_ai.thinking.underuse", unit="count")
tracer = trace.get_tracer("thinking-slo")


def record_thinking_usage(model: str, budget_tokens: int, used_tokens: int):
    """记录一次思考预算使用情况"""
    attrs = {"gen_ai.request.model": model}
    thinking_usage_hist.record(used_tokens, attributes=attrs)
    thinking_budget_hist.record(budget_tokens, attributes=attrs)
    utilization = used_tokens / max(budget_tokens, 1)
    with tracer.start_as_current_span(f"thinking.{model}") as span:
        span.set_attribute("gen_ai.thinking.utilization_ratio", round(utilization, 3))
        if utilization >= 1.0:
            thinking_overshoot_counter.add(1, attributes=attrs)
            span.set_attribute("gen_ai.thinking.slo_status", "overshoot")
        elif utilization < 0.3:
            thinking_waste_counter.add(1, attributes=attrs)
            span.set_attribute("gen_ai.thinking.slo_status", "waste")
        else:
            span.set_attribute("gen_ai.thinking.slo_status", "healthy")
    return utilization


if __name__ == "__main__":
    samples = [
        ("claude-sonnet-4-6", 2048, 512),    # healthy
        ("claude-sonnet-4-6", 2048, 2100),   # overshoot
        ("claude-sonnet-4-6", 2048, 200),    # waste
        ("claude-opus-4-6", 8192, 6000),     # healthy
        ("claude-haiku-4-5", 1024, 1100),    # overshoot
    ]
    for model, budget, used in samples:
        util = record_thinking_usage(model, budget, used)
        print(f"  {model}: util={util:.2f}")
    spans = exporter.get_finished_spans()
    statuses = [s.attributes.get("gen_ai.thinking.slo_status") for s in spans]
    print(f"status histogram: {dict((s, statuses.count(s)) for s in set(statuses))}")
