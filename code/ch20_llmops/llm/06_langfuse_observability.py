# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.3.3 LangFuse：开源可观测性替代方案
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langfuse (mocked fallback)
# run: python 06_langfuse_observability.py
# expected_runtime: < 1s
# expected_output: Mocked trace/spans created, response dict printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2033-langfuse开源可观测性替代方案-⭐⭐⭐⭐
# Interview hooks:
#  - LangFuse 与 LangSmith 的核心差异（开源 vs 商业）？
#  - @observe 装饰器如何自动构造 Span 树？
#  - 离线时如何 mock langfuse_context 行为？

import os
import uuid
from typing import Any, Dict, List

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    _HAS_LANGFUSE = bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))
except ImportError:
    Langfuse = None  # type: ignore
    observe = None  # type: ignore
    langfuse_context = None  # type: ignore
    _HAS_LANGFUSE = False


# ---------- 离线 mock 上下文 ----------
class _MockContext:
    def __init__(self):
        self.trace_attrs = {}
        self.observation_attrs = {}
        self.scores = {}

    def update_current_trace(self, **kwargs):
        self.trace_attrs.update(kwargs)

    def update_current_observation(self, **kwargs):
        self.observation_attrs.update(kwargs)

    def score_current_trace(self, name, value):
        self.scores[name] = value


def _noop_observe(*dargs, **dkwargs):
    if dargs and callable(dargs[0]) and not isinstance(dargs[0], type):
        return dargs[0]
    def _wrap(fn):
        return fn
    return _wrap


if not _HAS_LANGFUSE:
    observe = _noop_observe  # type: ignore
    _mock_ctx = _MockContext()

    class _MockCtxWrapper:
        def __getattr__(self, item):
            return getattr(_mock_ctx, item)

    langfuse_context = _MockCtxWrapper()  # type: ignore
    _mock_langfuse = type("LF", (), {"score": lambda *a, **k: None})()


@observe(name="customer-support-agent")
def handle_customer_query(query: str, conversation_history: List) -> Dict[str, Any]:
    """客户支持 Agent，LangFuse 自动追踪全链路"""
    langfuse_context.update_current_trace(
        name=f"support-{query[:30]}",
        tags=["production", "customer-support"],
        metadata={"user_tier": "premium", "channel": "web"},
    )

    intent = classify_intent(query)
    langfuse_context.update_current_observation(metadata={"intent": intent})
    docs = retrieve_knowledge(query, intent)
    answer = generate_response(query, docs, conversation_history)

    langfuse_context.score_current_trace(
        name="response_length",
        value=min(len(answer) / 500, 1.0),
    )
    return {"answer": answer, "intent": intent, "sources": len(docs)}


@observe()
def classify_intent(query: str) -> str:
    return "technical_support"


@observe()
def retrieve_knowledge(query: str, intent: str) -> List[str]:
    return ["doc1", "doc2"]


@observe()
def generate_response(query: str, docs: List, history: List) -> str:
    return "Generated answer..."


def evaluate_response_quality(trace_id: str, response: str):
    """异步评估回答质量（mocked）"""
    score = 0.85
    if _HAS_LANGFUSE:
        _mock_langfuse.score(
            trace_id=trace_id,
            name="quality_score",
            value=score,
            comment=f"Auto-evaluated: {score:.2f}",
        )
    print(f"[offline] trace={trace_id} quality_score={score}")


def main():
    result = handle_customer_query("我的订单还没到", [])
    print(result)
    evaluate_response_quality(f"trace-{uuid.uuid4().hex[:8]}", result["answer"])


if __name__ == "__main__":
    main()
    print("OK")
