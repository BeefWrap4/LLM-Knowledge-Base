# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.fastapi_prometheus
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: fastapi, prometheus_client, uvicorn (optional, for live serve)
# run: python 16_fastapi_prometheus.py
# expected_runtime: < 1s (builds FastAPI app; can be served with uvicorn)
# expected_output: Built FastAPI app with /metrics and /chat endpoints, without external calls
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - Counter / Histogram / Gauge 在 LLM 监控中分别适合记录什么？
#  - 延迟分桶 (buckets) 为什么对 LLM 推理尤其重要（0.1s ~ 30s）？
#  - /metrics 端点如何避免被业务流量耗尽？

import os

try:
    from fastapi import FastAPI
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
    from starlette.responses import Response

    _HAS_DEPS = True
except ImportError:
    FastAPI = None  # type: ignore
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    CollectorRegistry = None  # type: ignore
    CONTENT_TYPE_LATEST = None  # type: ignore
    generate_latest = None  # type: ignore
    Response = None  # type: ignore
    _HAS_DEPS = False


def build_app():
    """构建带 Prometheus 指标的 FastAPI 应用（仅在依赖存在时可用）。"""
    if not _HAS_DEPS:
        raise RuntimeError("fastapi + prometheus_client not installed")

    app = FastAPI()
    default_model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
    registry = CollectorRegistry()
    app.state.prometheus_registry = registry

    llm_requests_total = Counter(
        "llm_requests_total",
        "Total LLM requests",
        ["model", "status"],
        registry=registry,
    )
    llm_request_duration = Histogram(
        "llm_request_duration_seconds",
        "LLM request duration in seconds",
        ["model"],
        buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
        registry=registry,
    )
    llm_token_usage = Counter(
        "llm_token_usage_total",
        "Total tokens used",
        ["model", "type"],
        registry=registry,
    )

    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    @app.post("/chat")
    async def chat(request: dict):
        model = request.get("model", default_model)
        with llm_request_duration.labels(model=model).time():
            try:
                # 模拟 LLM 调用
                response = {"answer": "mock response"}
                llm_requests_total.labels(model=model, status="success").inc()
                llm_token_usage.labels(model=model, type="input").inc(100)
                llm_token_usage.labels(model=model, type="output").inc(50)
                return response
            except Exception:
                llm_requests_total.labels(model=model, status="error").inc()
                raise

    return app


if __name__ == "__main__":
    if not _HAS_DEPS:
        print("fastapi / prometheus_client not installed — skipping app build")
        print("Install: pip install fastapi prometheus_client uvicorn")
    else:
        app = build_app()
        print("FastAPI app built. Run with: uvicorn 16_fastapi_prometheus:app --reload")
        print("Available routes:")
        for r in app.routes:
            if hasattr(r, "path"):
                print(f"  {getattr(r, 'methods', '-')} {r.path}")
    print("OK")
