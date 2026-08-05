# ---
# chapter: 37
# topic: RAG、Agent 与安全评估
# topic_id: evaluation.phoenix_auto_instrument
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: arize-phoenix-otel, arize-phoenix-evals>=3, pandas
# run: python 13_phoenix_auto_instrument.py
# expected_runtime: <2s (mock mode)
# expected_output: Current Phoenix tracing/evaluation flow followed by OK
# ---
# See: ../../../37_RAG_Agent与安全评估.md
# Interview hooks:
# - What does OpenInference add on top of generic OpenTelemetry traces?
# - Why are tracing, evaluator input mapping, and score logging separate steps?
# - What fields does a FaithfulnessEvaluator require?

"""Phoenix OTel 注册与 ``arize-phoenix-evals>=3`` 客户端评估示例。

默认 ``LLM_MOCK=1``，不导入 SDK、不读取密钥、不连接 Phoenix 或模型服务。
"""

import os
from typing import Any

PHOENIX_MCP_CONFIG = """
{
  "mcpServers": {
    "phoenix": {
      "command": "npx",
      "args": ["-y", "@arizeai/phoenix-mcp"],
      "env": {
        "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:6006"
      }
    }
  }
}
""".strip()


def run_phoenix_demo() -> Any:
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[mock] Phoenix 当前流程（未连接 collector、未调用评审模型）")
        print("  register(...) -> OpenInference/OTel traces")
        print("  LLM(...) + FaithfulnessEvaluator(...)")
        print("  evaluate_dataframe(dataframe, evaluators)")
        print("  Client().spans.log_span_annotations_dataframe(...)")
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("真实模式需要 OPENAI_API_KEY；默认请使用 LLM_MOCK=1")

    try:
        import pandas as pd
        from phoenix.evals import LLM, evaluate_dataframe
        from phoenix.evals.metrics import FaithfulnessEvaluator
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError("真实模式需要 arize-phoenix-otel、arize-phoenix-evals>=3 和 pandas") from exc

    register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "ch17-evaluation"),
        endpoint=os.environ.get(
            "PHOENIX_OTLP_ENDPOINT",
            "http://localhost:6006/v1/traces",
        ),
        set_global_tracer_provider=True,
    )

    dataframe = pd.DataFrame(
        [
            {
                "input": "CPython 中的 GIL 是什么？",
                "output": "GIL 是 CPython 的全局解释器锁。",
                "context": "GIL 是 CPython 解释器用于协调 Python 字节码执行的全局锁。",
            }
        ]
    )
    judge = LLM(
        provider="openai",
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        client="openai",
    )
    evaluator = FaithfulnessEvaluator(llm=judge)
    result = evaluate_dataframe(dataframe=dataframe, evaluators=[evaluator])
    print(result)
    return result


if __name__ == "__main__":
    run_phoenix_demo()
    print("OK")
