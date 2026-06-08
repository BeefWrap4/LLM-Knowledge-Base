import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.11.2 Langfuse v3 评估与可观测性平台
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langfuse, openai
# run: python 12_langfuse_v3.py
# expected_runtime: <2s (mock mode)
# expected_output: Demonstrates Langfuse prompt management + LLM-as-Judge skeleton
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What changed in Langfuse v3 (ClickHouse + OTel) vs v2?
# - Why is centralized prompt management critical for production LLM apps?
# - How does Langfuse integrate with the broader OpenTelemetry ecosystem?

"""Langfuse v3 评估与 Prompt 管理示例。

Langfuse v3 将 LLM 可观测性、Prompt 管理、评估、LLM-as-Judge 整合到同一平台，
从 v2 的 PostgreSQL 单一后端演进为 ClickHouse 事件存储 + OTel 协议。
"""
import os


def run_langfuse_v3_demo() -> None:
    mock_mode = os.environ.get("LANGFUSE_MOCK", "1") == "1"

    if mock_mode:
        print("[mock] Langfuse v3 + Prompt 管理 + LLM-as-Judge 流程")
        print("[mock] 1. 初始化 Langfuse 客户端 (cloud.langfuse.com)")
        print("[mock] 2. 拉取 prompt='summarizer', version=3, label='production'")
        print("[mock] 3. 调用 GPT-4o 生成 summary")
        print("[mock] 4. OTel 自动上报 token/latency/model-args")
        print("[mock] 5. 跑 LLM-as-Judge 'relevance' (threshold=0.7)")
        print("[mock] 评估结果: passes=True")
        return

    try:
        from langfuse import Langfuse
        from langfuse.decorators import langfuse_context, observe
        from langfuse.evaluation import evaluate

        from shared.llm_client import UnifiedClient  # Wave 16
    except ImportError as exc:
        print(f"[mock] langfuse/openai 未安装 ({exc})，使用模拟输出")
        return

    # 1. 初始化 Langfuse v3
    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-..."),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    openai_client = UnifiedClient()  # Wave 16: 统一多厂商 (deepseek/kimi/siliconflow/MiniMax)

    # 2. 集中式 Prompt 管理（生产环境不写死在代码里）
    prompt = langfuse.get_prompt("summarizer", version=3, label="production")
    compiled_prompt = prompt.compile(variables={"max_words": 200})

    # 3. 追踪 LLM 调用（OpenTelemetry 自动捕获）
    @observe(as_type="generation")
    def summarize(text: str) -> str:
        resp = openai_client.chat(
            messages=[{"role": "user", "content": f"{compiled_prompt}\n\n{text}"}],
        )
        # 自动上报 token / 延迟 / 模型参数
        langfuse_context.update_current_observation(
            model=resp.model,
            usage={
                "input": resp.usage["prompt_tokens"],
                "output": resp.usage["completion_tokens"],
            },
        )
        return resp.choices[0].message.content

    # 4. LLM-as-Judge 评估（内置 60+ 模板）
    judge_prompt = langfuse.get_prompt("judge-summarization", version=1)
    sample = "原文本"
    result = evaluate(
        data=[{"input": sample, "output": summarize(sample)}],
        evaluators=[
            {
                "name": "relevance",
                "prompt": judge_prompt.compile(),
                "model": "gpt-4o",
                "threshold": 0.7,
            }
        ],
    )
    print(f"评估结果: {result.passes}")


if __name__ == "__main__":
    run_langfuse_v3_demo()
