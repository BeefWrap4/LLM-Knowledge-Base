# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.langsmith_observability
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langsmith, openai (mocked fallback)
# run: python 05_langsmith_observability.py
# expected_runtime: < 1s (mocked) / depends on API (live)
# expected_output: Trace/Run created, mock QA pipeline output printed
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - LangSmith 的 Trace / Run / Feedback 三层抽象如何映射到 LLM 应用？
#  - 在没有 LangSmith 账户时如何离线模拟其关键数据流？
#  - 用 @traceable 装饰器追踪函数时，输入/输出是如何序列化的？

import os
import uuid
from typing import Any

_LIVE_OBSERVABILITY = (
    os.environ.get("LLM_REAL_API") == "1" and os.environ.get("LLM_MOCK") == "0"
)
try:
    if not _LIVE_OBSERVABILITY:
        raise ImportError
    from langsmith import traceable

    _HAS_LANGSMITH = True
except ImportError:
    traceable = None  # type: ignore
    _HAS_LANGSMITH = False


# 离线装饰器兜底
def _noop_traceable(*dargs, **dkwargs):
    """离线 mock：保持签名一致，无副作用。"""
    if dargs and callable(dargs[0]) and not dargs[0].__name__.startswith("_"):
        return dargs[0]

    def _wrap(fn):
        return fn

    return _wrap


if not _HAS_LANGSMITH:
    traceable = _noop_traceable  # type: ignore


@traceable(run_type="chain", name="QA Pipeline", metadata={"version": "1.2.0", "environment": "staging"})
def qa_pipeline(question: str, context_docs: list[str]) -> dict[str, Any]:
    """完整的 QA 流水线，LangSmith 自动追踪每个步骤"""
    prompt = build_prompt(question, context_docs)
    answer = call_llm(prompt)
    result = post_process(answer)
    return result


@traceable(run_type="prompt", name="Build Prompt")
def build_prompt(question: str, docs: list[str]) -> str:
    """构建 Prompt（LangSmith 自动记录输入/输出）"""
    context = "\n\n".join(docs)
    return f"""基于以下上下文回答问题。

上下文：
{context}

问题：{question}

回答："""


@traceable(run_type="llm", name="LLM Call")
def call_llm(prompt: str) -> str:
    """离线 stub；真实 SDK 集成需显式上报 Provider usage，不能假定自动得到 Token。"""
    return f"[mocked answer] {prompt[:80]}"


@traceable(run_type="chain", name="Post-process")
def post_process(answer: str) -> dict[str, Any]:
    """后处理"""
    return {
        "answer": answer.strip(),
        "length": len(answer),
        "has_citation": "来源" in answer,
    }


def manual_trace_example():
    """手动创建 Trace 并添加 Feedback（离线版）"""
    run_id = f"offline-{uuid.uuid4().hex[:8]}"
    print(f"[offline] create_run id={run_id}")
    result = qa_pipeline("什么是 MCP 协议？", ["MCP (Model Context Protocol) 是 Anthropic 推出的..."])
    print(f"[offline] update_run outputs for id={run_id}")
    print(
        f"[offline] create_feedback user-rating=0.9, contains-citation={1.0 if result['has_citation'] else 0.0}"
    )


def main():
    if not _HAS_LANGSMITH:
        print("LangSmith live tracing disabled — running offline data flow")
    result = qa_pipeline("什么是 MCP 协议？", ["MCP (Model Context Protocol) 是 Anthropic 推出的..."])
    print(f"Answer: {result['answer'][:100]}...")
    if _HAS_LANGSMITH:
        print("🔗 在 LangSmith UI 查看完整 Trace")
    manual_trace_example()
    print("OK")


if __name__ == "__main__":
    main()
