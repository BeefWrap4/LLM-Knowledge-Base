# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.4 DeepEval
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: deepeval
# run: python 08_deepeval_rag.py
# expected_runtime: <2s (mock mode) / 10-30s (real)
# expected_output: Per-metric scores for the RAG test case
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Why is DeepEval considered "pytest for LLMs"?
# - Compare HallucinationMetric (deepeval) vs Faithfulness (ragas).
# - How does DeepEval integrate with CI/CD pipelines?

"""DeepEval 评估示例。

DeepEval 是一个面向 LLM 应用的测试框架，类似于"LLM 的 pytest"。
"""
import os


def run_deepeval_test() -> None:
    mock_mode = os.environ.get("DEEPEVAL_MOCK", "1") == "1"

    if mock_mode:
        print("[mock] DeepEval RAG 评估输出")
        print("AnswerRelevancyMetric: 0.875")
        print("FaithfulnessMetric: 0.920")
        print("ContextualRecallMetric: 0.850")
        print("ContextualPrecisionMetric: 0.830")
        print("HallucinationMetric: 0.150")
        return

    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualRecallMetric,
            ContextualPrecisionMetric,
            HallucinationMetric,
        )
    except ImportError as exc:
        print(f"[mock] deepeval 未安装 ({exc})，使用模拟输出")
        return

    # 创建测试用例
    test_case = LLMTestCase(
        input="什么是机器学习中的过拟合？",
        actual_output=(
            "过拟合是指模型在训练数据上表现很好，但在测试数据上表现差的现象。"
            "这通常是因为模型过于复杂，学习了训练数据中的噪声而非真正的模式。"
            "解决方法包括：正则化（L1/L2）、Dropout、早停、数据增强、交叉验证等。"
        ),
        expected_output="过拟合是模型对训练数据学习过度，导致泛化能力差的现象。",
        retrieval_context=[
            "过拟合发生在模型在训练数据上表现良好但泛化到新数据时表现不佳的情况。",
            "正则化和增加训练数据是解决过拟合的常用方法。",
            "过拟合与模型的复杂度有关，过于复杂的模型更容易过拟合。",
        ],
    )

    # 定义评估指标
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
        HallucinationMetric(threshold=0.3),  # 幻觉分数越低越好
    ]

    # 执行测试
    for metric in metrics:
        metric.measure(test_case)
        print(f"{metric.__class__.__name__}: {metric.score:.3f}")

    # 显式 assert 也可以用 assert_test
    assert_test(test_case, metrics)


if __name__ == "__main__":
    run_deepeval_test()
    print("OK")
