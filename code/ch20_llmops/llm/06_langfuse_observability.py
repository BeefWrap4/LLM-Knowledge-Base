# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.3.3 Langfuse：开源可观测性平台
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langfuse>=4 (mocked fallback)
# run: python 06_langfuse_observability.py
# expected_runtime: < 1s
# expected_output: Offline observation tree and score flow printed; no key read or network request
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2033-langfuse开源可观测性替代方案-⭐⭐⭐⭐
# Interview hooks:
#  - Langfuse 与 LangSmith 应按哪些部署、治理和集成维度选型？
#  - Python SDK v4 的 observe / get_client / propagate_attributes 如何构造观察树？
#  - 为什么默认关闭输入输出捕获，评分又要区分当前 Trace 与事后 Trace？

import os
import uuid
from contextlib import contextmanager
from typing import Any

_LIVE_OBSERVABILITY = (
    os.environ.get("LLM_REAL_API") == "1" and os.environ.get("LLM_MOCK") == "0"
)

try:
    if not _LIVE_OBSERVABILITY:
        raise ImportError
    from langfuse import get_client, observe, propagate_attributes

    _HAS_LANGFUSE = True
except ImportError:
    get_client = None  # type: ignore
    observe = None  # type: ignore
    propagate_attributes = None  # type: ignore
    _HAS_LANGFUSE = False


class _OfflineLangfuseClient:
    """只保留本例会用到的 v4 客户端表面；不读取凭据，也不联网。"""

    def __init__(self):
        self.trace_id = uuid.uuid4().hex
        self.span_updates: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []

    def get_current_trace_id(self) -> str:
        return self.trace_id

    def update_current_span(self, **kwargs):
        self.span_updates.append(kwargs)

    def score_current_trace(self, **kwargs):
        self.scores.append({"trace_id": self.trace_id, **kwargs})

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        return None


def _noop_observe(*decorator_args, **decorator_kwargs):
    del decorator_kwargs
    if decorator_args and callable(decorator_args[0]):
        return decorator_args[0]

    def _wrap(function):
        return function

    return _wrap


@contextmanager
def _noop_propagate_attributes(**kwargs):
    del kwargs
    yield


if _HAS_LANGFUSE:
    _langfuse_client = get_client()
else:
    observe = _noop_observe  # type: ignore
    propagate_attributes = _noop_propagate_attributes  # type: ignore
    _langfuse_client = _OfflineLangfuseClient()


@observe(
    name="customer-support-agent",
    as_type="agent",
    capture_input=False,
    capture_output=False,
)
def handle_customer_query(query: str, conversation_history: list) -> dict[str, Any]:
    """构造 v4 observation tree；默认不把原始问答写入遥测。"""
    with propagate_attributes(
        trace_name="customer-support-agent",
        tags=["customer-support"],
        metadata={"user_tier": "premium", "channel": "web"},
    ):
        intent = classify_intent(query)
        _langfuse_client.update_current_span(metadata={"intent": intent})
        docs = retrieve_knowledge(query, intent)
        answer = generate_response(query, docs, conversation_history)
        safe_output = {"intent": intent, "source_count": len(docs)}
        _langfuse_client.update_current_span(output=safe_output)
        _langfuse_client.score_current_trace(
            name="response_length",
            value=min(len(answer) / 500, 1.0),
            data_type="NUMERIC",
        )
        trace_id = _langfuse_client.get_current_trace_id()

    return {
        "answer": answer,
        "intent": intent,
        "sources": len(docs),
        "trace_id": trace_id,
    }


@observe(as_type="span", capture_input=False, capture_output=False)
def classify_intent(query: str) -> str:
    del query
    return "technical_support"


@observe(as_type="retriever", capture_input=False, capture_output=False)
def retrieve_knowledge(query: str, intent: str) -> list[str]:
    del query, intent
    return ["doc1", "doc2"]


@observe(as_type="span", capture_input=False, capture_output=False)
def generate_response(query: str, docs: list, history: list) -> str:
    del query, docs, history
    return "Generated answer..."


def evaluate_response_quality(trace_id: str, response: str):
    """用 v4 create_score 对已结束的 Trace 添加异步评分。"""
    score = 0.85
    _langfuse_client.create_score(
        trace_id=trace_id,
        name="quality_score",
        value=score,
        data_type="NUMERIC",
        comment=f"Auto-evaluated: {score:.2f}",
    )
    print(f"[{'live' if _HAS_LANGFUSE else 'offline'}] trace={trace_id} quality_score={score}")


def main():
    if not _HAS_LANGFUSE:
        print("Langfuse live tracing disabled — running offline v4-shaped data flow")
    result = handle_customer_query("我的订单还没到", [])
    print({key: value for key, value in result.items() if key != "trace_id"})
    evaluate_response_quality(result["trace_id"], result["answer"])
    _langfuse_client.flush()
    print("OK")


if __name__ == "__main__":
    main()
