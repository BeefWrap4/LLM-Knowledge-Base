# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.3 OpenTelemetry SDK + GenAI 语义约定
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc
# run: python 19_otel_genai_telemetry.py
# expected_runtime: < 1s (no live exporter; uses InMemorySpanExporter)
# expected_output: Span dict with gen_ai.* attributes printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20103-opentelemetry-sdk--genai-语义约定
# Interview hooks:
#  - 为什么 2026 年大厂面试越来越要求 OTel GenAI SemConv？
#  - gen_ai.usage.cached_input_tokens 与 cached_input_tokens 字段如何区分？
#  - OTLP gRPC 与 HTTP exporter 的取舍（吞吐 vs 兼容性）？

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider


def build_provider():
    resource = Resource.create(
        {
            SERVICE_NAME: "qa-agent-prod",
            SERVICE_VERSION: "v2.3.0",
            "gen_ai.system": "openai",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    metrics.set_meter_provider(MeterProvider(resource=resource))
    return provider, exporter


class GenAITelemetry:
    def __init__(self, tracer, meter):
        self.tracer = tracer
        self.meter = meter
        self.cost_hist = meter.create_histogram("gen_ai.cost.usd", unit="usd")
        self.input_tokens = meter.create_counter("gen_ai.usage.input_tokens", unit="tokens")
        self.output_tokens = meter.create_counter("gen_ai.usage.output_tokens", unit="tokens")

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        finish_reason: str = "stop",
        cached_input_tokens: int = 0,
        tool_calls=None,
        retrieval=None,
        judge_scores=None,
        user_id: str | None = None,
        trajectory_id: str | None = None,
    ):
        with self.tracer.start_as_current_span(
            f"chat {model}", kind=trace.SpanKind.CLIENT
        ) as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.request.temperature", 0.7)
            span.set_attribute("gen_ai.request.max_tokens", 2048)
            span.set_attribute("gen_ai.response.model", model)
            span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if cached_input_tokens:
                span.set_attribute("gen_ai.usage.cached_input_tokens", cached_input_tokens)
            span.set_attribute("gen_ai.cost.usd", cost_usd)

            for idx, tc in enumerate(tool_calls or []):
                span.set_attribute(f"gen_ai.tool.name.{idx}", tc.get("name", ""))
                span.set_attribute(f"gen_ai.tool.call.id.{idx}", tc.get("id", ""))

            if retrieval:
                span.set_attribute("gen_ai.retrieval.hit", retrieval.get("hit", False))
                span.set_attribute("gen_ai.retrieval.documents", retrieval.get("documents", 0))

            if judge_scores:
                for name, score in judge_scores.items():
                    span.set_attribute(f"gen_ai.evaluation.{name}", score)

            if user_id:
                span.set_attribute("enduser.id", user_id)
            if trajectory_id:
                span.set_attribute("gen_ai.agent.trajectory_id", trajectory_id)

            common_attrs = {"gen_ai.system": "openai", "gen_ai.response.model": model}
            self.cost_hist.record(cost_usd, attributes=common_attrs)
            self.input_tokens.add(input_tokens, attributes=common_attrs)
            self.output_tokens.add(output_tokens, attributes=common_attrs)


if __name__ == "__main__":
    provider, exporter = build_provider()
    tracer = trace.get_tracer("qa-agent.instrumentation", "1.0.0")
    meter = metrics.get_meter("qa-agent.metrics")
    telemetry = GenAITelemetry(tracer, meter)

    telemetry.record_llm_call(
        model="claude-sonnet-4-6",
        input_tokens=128,
        output_tokens=87,
        cost_usd=0.001342,
        finish_reason="end_turn",
        cached_input_tokens=64,
        tool_calls=[{"name": "search_web", "id": "toolu_01A"}],
        retrieval={"hit": True, "documents": 5},
        judge_scores={"relevance": 0.92, "factuality": 0.88},
        user_id="u_12345",
        trajectory_id="traj-abc-001",
    )

    spans = exporter.get_finished_spans()
    print(f"recorded {len(spans)} span(s)")
    for s in spans:
        print(f"name={s.name}, attrs={dict(s.attributes)}")
    print("OK")
