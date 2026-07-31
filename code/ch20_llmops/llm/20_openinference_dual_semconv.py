# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.4 OpenInference + OpenTelemetry 双规范导出
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry, openinference-instrumentation-langchain, openinference-instrumentation-openai
# run: python 20_openinference_dual_semconv.py
# expected_runtime: < 1s (mostly demonstrates the API; no live exporter)
# expected_output: Configured TracerProvider with dual-semconv attribute mapping
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20104-openinference--opentelemetry-双规范导出
# Interview hooks:
#  - OpenInference 与 OTel GenAI SemConv 是替代还是互补关系？
#  - RETRIEVER / TOOL / AGENT / LLM / CHAIN 几个 SpanKind 在 RAG/Agent 场景下如何选择？
#  - 双规范导出时如何避免属性命名冲突（属性重命名 / prefix）？

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# 1. 初始化 OTel Provider
resource = Resource.create({SERVICE_NAME: "rag-qa-service", SERVICE_VERSION: "1.4.0"})
provider = TracerProvider(resource=resource)
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# 2. 自动 instrument 只在显式 live 模式启用；离线验收不改写 SDK 客户端。
import os

try:
    if os.environ.get("LLM_REAL_API") != "1" or os.environ.get("LLM_MOCK") != "0":
        raise ImportError
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=provider)
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    _HAS_OI = True
except Exception:
    _HAS_OI = False


# OpenInference SpanKind 枚举（mocked）
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
SPAN_KIND_VALUES = {
    "CHAIN",
    "LLM",
    "RETRIEVER",
    "TOOL",
    "AGENT",
    "EMBEDDING",
    "RERANKER",
    "UNKNOWN",
}


def instrument_retrieval(query: str, top_k: int = 5, capture_content: bool = False):
    """手动记录 RAG 检索 Span（OpenInference RETRIEVER Kind）"""
    tracer = trace.get_tracer("rag.retriever")
    with tracer.start_as_current_span("vector_search") as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "RETRIEVER")
        if capture_content:
            span.set_attribute("retrieval.query.text", query)
        span.set_attribute("retrieval.top_k", top_k)

        # 业务执行（离线 mock）
        docs = [
            type(
                "Doc",
                (),
                {
                    "metadata": {"id": f"doc_{i}", "score": 0.9 - i * 0.1},
                    "page_content": f"document content {i}",
                },
            )()
            for i in range(min(top_k, 3))
        ]

        span.set_attribute("retrieval.document.count", len(docs))
        for i, d in enumerate(docs[:3]):
            span.set_attribute(f"retrieval.documents.{i}.document.id", d.metadata["id"])
            span.set_attribute(f"retrieval.documents.{i}.document.score", d.metadata["score"])
            if capture_content:
                span.set_attribute(f"retrieval.documents.{i}.document.content", d.page_content[:256])

        span.set_attribute("app.rag.hit", len(docs) > 0)
        if docs:
            span.set_attribute(
                "app.rag.score_max",
                max(d.metadata["score"] for d in docs),
            )
        return docs


def instrument_agent_step(
    step_name: str,
    decision: str,
    observation: str,
    capture_content: bool = False,
):
    """手动记录 Agent 决策 Span（OpenInference AGENT Kind）"""
    tracer = trace.get_tracer("agent.react")
    with tracer.start_as_current_span(f"agent.{step_name}") as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "AGENT")
        span.set_attribute("agent.step.name", step_name)
        if capture_content:
            span.set_attribute("agent.decision", decision)
            span.set_attribute("agent.observation", observation[:512])
        return decision


if __name__ == "__main__":
    print(f"openinference instrumentation: {'on' if _HAS_OI else 'off (mocked)'}")
    instrument_retrieval("量子纠缠", top_k=3)
    instrument_agent_step("plan", "search_web", "found 3 docs")
    spans = exporter.get_finished_spans()
    for s in spans:
        print(f"  span={s.name}, kind={s.attributes.get(OPENINFERENCE_SPAN_KIND)}")
    print("OK")
