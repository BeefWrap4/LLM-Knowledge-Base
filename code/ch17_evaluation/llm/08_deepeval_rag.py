# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.4 DeepEval
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: deepeval
# run: python 08_deepeval_rag.py
# expected_runtime: <2s (mock mode) / workload-dependent (real)
# expected_output: Offline metric configuration or measured values followed by OK
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Why is DeepEval considered pytest for LLM applications?
# - Why is Faithfulness preferable to HallucinationMetric for a RAG test case?
# - How do explicit thresholds become CI quality gates?

"""DeepEval RAG 评估示例。

默认 ``LLM_MOCK=1``，不导入 DeepEval、不读取密钥且不联网。显式设置
``LLM_MOCK=0`` 后才创建评审模型；任何真实模式错误都会向上抛出。
"""

import os
from typing import Any


def run_deepeval_test() -> list[Any]:
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[mock] DeepEval RAG 指标配置；未调用评审模型、未生成分数")
        print(
            "  AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric, ContextualPrecisionMetric"
        )
        return []

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("真实模式需要 OPENAI_API_KEY；默认请使用 LLM_MOCK=1")

    try:
        from deepeval import assert_test
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.models import GPTModel
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError("真实模式需要 deepeval：pip install deepeval") from exc

    judge_model = GPTModel(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        generation_kwargs={"reasoning_effort": "low"},
    )
    test_case = LLMTestCase(
        input="什么是机器学习中的过拟合？",
        actual_output=(
            "过拟合是模型在训练数据上表现好、在未见数据上泛化差的现象。"
            "常见缓解方法包括正则化、早停、数据增强和交叉验证。"
        ),
        expected_output="过拟合是模型对训练数据学习过度，导致泛化能力差的现象。",
        retrieval_context=[
            "过拟合表现为训练集性能良好，但对未见数据的泛化性能不佳。",
            "正则化和增加有效训练数据是常见缓解方法。",
        ],
    )
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        ContextualRecallMetric(threshold=0.7, model=judge_model),
        ContextualPrecisionMetric(threshold=0.7, model=judge_model),
    ]

    for metric in metrics:
        metric.measure(test_case)
        print(f"{metric.__class__.__name__}: {metric.score:.3f}")
    assert_test(test_case, metrics)
    return metrics


if __name__ == "__main__":
    run_deepeval_test()
    print("OK")
