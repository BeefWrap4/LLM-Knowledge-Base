# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.3 TruLens RAG 三元组
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: trulens, trulens-providers-openai
# run: python 07_trulens_rag_triad.py
# expected_runtime: <2s (mock mode)
# expected_output: Current TruLens RAG Triad metric definitions followed by OK
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What is the RAG Triad and which failure layer does each metric diagnose?
# - Why should context-level scores be aggregated explicitly?
# - How do TruLens selectors connect traces to evaluation inputs?

"""使用 TruLens 当前 ``Metric``/``Selector`` API 定义 RAG Triad。

默认 ``LLM_MOCK=1`` 只展示配置，不导入 SDK、读取密钥或联网。
"""

import os
from statistics import mean
from typing import Any


def setup_trulens_rag_triad() -> list[Any]:
    """定义三项指标；真实采集时再把它们传给 ``TruApp``。"""
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[mock] TruLens Metric/Selector RAG Triad 配置（未调用评审模型）")
        print("  - Groundedness: context -> answer")
        print("  - Answer Relevance: question -> answer")
        print("  - Context Relevance: question -> each context, then aggregate")
        return []

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("真实模式需要 OPENAI_API_KEY；默认请使用 LLM_MOCK=1")

    try:
        from trulens.core import Metric, Selector
        from trulens.providers.openai import OpenAI as OpenAIFeedback
    except ImportError as exc:
        raise RuntimeError("真实模式需要 trulens 和 trulens-providers-openai") from exc

    provider = OpenAIFeedback(model_engine=os.environ.get("OPENAI_MODEL", "gpt-5.6"))
    groundedness = Metric(
        implementation=(provider.groundedness_measure_with_cot_reasons_consider_answerability),
        name="Groundedness",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
            "question": Selector.select_record_input(),
        },
    )
    answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )
    context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="Context Relevance",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=mean,
    )

    metrics = [groundedness, answer_relevance, context_relevance]
    for metric in metrics:
        print(f"defined: {metric.name}")
    return metrics


if __name__ == "__main__":
    setup_trulens_rag_triad()
    print("OK")
