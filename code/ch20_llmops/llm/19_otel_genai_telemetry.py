# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.3 OpenTelemetry SDK + GenAI 语义约定
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc
# run: python 19_otel_genai_telemetry.py
# expected_runtime: < 1s (no live exporter; uses in-memory readers)
# expected_output: Chat/tool Span attributes + gen_ai.client.token.usage metric names
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20103-opentelemetry-sdk--genai-语义约定
# Interview hooks:
#  - OTel GenAI SemConv 如何降低跨 SDK、跨后端的字段映射成本？
#  - gen_ai.usage.cache_read.input_tokens 与总 input_tokens 如何区分？
#  - OTLP gRPC 与 HTTP exporter 应如何按网络、代理、运维和实测性能选型？

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def build_provider():
    resource = Resource.create(
        {
            SERVICE_NAME: "qa-agent-prod",
            SERVICE_VERSION: "v2.3.0",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    metric_reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))
    return provider, exporter, metric_reader


class GenAITelemetry:
    def __init__(self, tracer, meter):
        self.tracer = tracer
        self.meter = meter
        self.cost_hist = meter.create_histogram("app.llm.cost", unit="usd")
        self.token_usage = meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
        )

    def record_llm_call(
        self,
        provider_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
        response_id: str | None = None,
        finish_reason: str = "stop",
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        tool_calls=None,
        retrieval=None,
        judge_scores=None,
        user_id: str | None = None,
        trajectory_id: str | None = None,
    ):
        with self.tracer.start_as_current_span(f"chat {model}", kind=trace.SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.provider.name", provider_name)
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.request.temperature", 0.7)
            span.set_attribute("gen_ai.request.max_tokens", 2048)
            span.set_attribute("gen_ai.response.model", model)
            if response_id:
                span.set_attribute("gen_ai.response.id", response_id)
            span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if cached_input_tokens:
                span.set_attribute("gen_ai.usage.cache_read.input_tokens", cached_input_tokens)
            if cache_creation_input_tokens:
                span.set_attribute(
                    "gen_ai.usage.cache_creation.input_tokens",
                    cache_creation_input_tokens,
                )
            if cost_usd is not None:
                span.set_attribute("app.llm.cost.usd", cost_usd)

            for tool_call in tool_calls or []:
                tool_name = tool_call.get("name", "unknown")
                with self.tracer.start_as_current_span(
                    f"execute_tool {tool_name}",
                    kind=trace.SpanKind.INTERNAL,
                ) as tool_span:
                    tool_span.set_attribute("gen_ai.operation.name", "execute_tool")
                    tool_span.set_attribute("gen_ai.tool.name", tool_name)
                    tool_span.set_attribute("gen_ai.tool.call.id", tool_call.get("id", ""))
                    # Arguments/results are opt-in content and may contain secrets or PII.

            if retrieval:
                span.set_attribute("app.rag.hit", retrieval.get("hit", False))
                span.set_attribute("app.rag.retrieved_documents", retrieval.get("documents", 0))

            if judge_scores:
                for name, score in judge_scores.items():
                    span.set_attribute(f"app.evaluation.{name}", score)

            if user_id:
                span.set_attribute("user.id", user_id)
            if trajectory_id:
                span.set_attribute("app.agent.trajectory_id", trajectory_id)

            common_attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider_name,
                "gen_ai.request.model": model,
                "gen_ai.response.model": model,
            }
            if cost_usd is not None:
                self.cost_hist.record(cost_usd, attributes=common_attrs)
            self.token_usage.record(
                input_tokens,
                attributes={**common_attrs, "gen_ai.token.type": "input"},
            )
            self.token_usage.record(
                output_tokens,
                attributes={**common_attrs, "gen_ai.token.type": "output"},
            )


if __name__ == "__main__":
    provider, exporter, metric_reader = build_provider()
    tracer = trace.get_tracer("qa-agent.instrumentation", "1.0.0")
    meter = metrics.get_meter("qa-agent.metrics")
    telemetry = GenAITelemetry(tracer, meter)

    observed_cost = os.environ.get("LLM_OBSERVED_COST_USD")
    telemetry.record_llm_call(
        provider_name=os.environ.get("LLM_PROVIDER_NAME", "anthropic"),
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
        input_tokens=128,
        output_tokens=87,
        cost_usd=float(observed_cost) if observed_cost is not None else None,
        response_id="offline-response-001",
        finish_reason="end_turn",
        cached_input_tokens=64,
        cache_creation_input_tokens=0,
        tool_calls=[{"name": "search_web", "id": "toolu_01A"}],
        retrieval={"hit": True, "documents": 5},
        judge_scores={"relevance": 0.92, "factuality": 0.88},
        user_id=None,
        trajectory_id="traj-abc-001",
    )

    spans = exporter.get_finished_spans()
    print(f"recorded {len(spans)} span(s)")
    for s in spans:
        print(f"name={s.name}, attrs={dict(s.attributes)}")

    metric_names = [
        metric.name
        for resource_metrics in metric_reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    ]
    print(f"metrics={metric_names}")
    print("OK")
