# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.11.3 Phoenix Auto-Instrumentation
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: arize-phoenix, phoenix.evals
# run: python 13_phoenix_auto_instrument.py
# expected_runtime: <2s (mock mode)
# expected_output: Phoenix OTel + offline batch eval skeleton
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What is the OpenInference protocol and why does it matter?
# - How does Phoenix's "auto-instrumentation" differ from manual tracing?
# - How would you use the SpanQuery DSL to drill into specific failure modes?

"""Phoenix MCP Server + Auto-Instrumentation 示例。

Phoenix（Arize AI 开源）是 LLM 可观测性领域的标杆项目，
以 OpenInference 协议为核心，强调 Auto-Instrumentation（零代码一行启动）。
"""
import os

# ---- Phoenix MCP Server 配置（Claude / Cursor 直连 Phoenix 数据）----
# 配置文件：~/.config/claude/claude_desktop_config.json
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
"""


def run_phoenix_demo() -> None:
    print("=== Phoenix MCP Server 配置 (JSON) ===")
    print(PHOENIX_MCP_CONFIG)

    mock_mode = os.environ.get("PHOENIX_MOCK", "1") == "1"
    if mock_mode:
        print("[mock] Phoenix Auto-Instrumentation 流程")
        print("[mock] tracer_provider = register(project_name='my-llm-app', ...)")
        print("[mock] 之后所有 OpenAI / LangChain 调用自动 trace，零侵入")
        print("[mock] 用 SpanQuery DSL 查询 span: child_of='retrieval'")
        print("[mock] HallucinationEvaluator(model='gpt-4o') -> run_evals")
        print("[mock] 结果 DataFrame head: 3 rows × 4 cols")
        return

    try:
        from phoenix.otel import register
    except ImportError as exc:
        print(f"[mock] phoenix 未安装 ({exc})，使用模拟输出")
        return

    # ---- Python 端：Phoenix Auto-Instrumentation ----
    # 一行代码启动 OpenTelemetry trace
    tracer_provider = register(
        project_name="my-llm-app",
        endpoint="http://localhost:6006/v1/traces",
        set_global_tracer_provider=True,
    )
    # 之后所有 OpenAI / LangChain 调用自动 trace，零侵入

    # ---- 离线批量评估 ----
    try:
        from phoenix.evals import HallucinationEvaluator, run_evals
        from phoenix.trace.dsl import SpanQuery
        import phoenix as px
    except ImportError as exc:
        print(f"[mock] phoenix.evals 未安装 ({exc})")
        return

    # 用 DSL 查询特定的 span
    query = (
        SpanQuery()
        .select(
            span="llm.generation",
            child_of="retrieval",
            columns=["input.value", "output.value", "context.value"],
        )
    )

    spans_df = px.Client().query_spans(query)

    # 跑幻觉评估（内置 LLM judge）
    hallucination_eval = HallucinationEvaluator(model="gpt-4o")
    hallucination_results = run_evals(
        dataframe=spans_df,
        evaluators=[hallucination_eval],
    )
    print(hallucination_results.head())


if __name__ == "__main__":
    run_phoenix_demo()
    print("OK")
