# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.11.4 DeepEval DAG + G-Eval
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: deepeval
# run: python 14_deepeval_dag_geval.py
# expected_runtime: <2s (mock mode) / workload-dependent (real)
# expected_output: Current DAG/G-Eval configuration followed by OK
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - When is a DAG rubric preferable to one holistic G-Eval criterion?
# - How are terminal DAG scores normalized?
# - Why can G-Eval require a different judge model from other metrics?

"""DeepEval 当前 DAG 与 G-Eval API 示例。

常规指标和 DAG 使用 ``OPENAI_MODEL``（默认 ``gpt-5.6``）。DeepEval 当前文档
说明 G-Eval 依赖 token log probabilities，因此单独用 ``DEEPEVAL_GEVAL_MODEL``
（默认 ``gpt-5.4``，可按最新兼容列表覆盖）。默认 mock 不导入 SDK 或联网。
"""

import os
from typing import Any


def run_deepeval_dag_geval() -> list[Any]:
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[mock] DeepEval 当前 DAG + G-Eval 配置（未调用评审模型）")
        print("  BinaryJudgementNode: task complete? false->0, true->groundedness")
        print("  BinaryJudgementNode: grounded? false->3, true->10")
        print("  DeepAcyclicGraph(root_nodes=[...]) -> DAGMetric")
        print("  GEval uses a separately configurable logprob-capable judge")
        return []

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("真实模式需要 OPENAI_API_KEY；默认请使用 LLM_MOCK=1")

    try:
        from deepeval import assert_test
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            DAGMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph
        from deepeval.models import GPTModel
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise RuntimeError("真实模式需要 deepeval：pip install deepeval") from exc

    standard_judge = GPTModel(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        generation_kwargs={"reasoning_effort": "low"},
    )
    # DeepEval G-Eval 需要 judge 暴露 log probabilities；兼容性应随版本复核。
    geval_judge = GPTModel(model=os.environ.get("DEEPEVAL_GEVAL_MODEL", "gpt-5.4"))
    correctness_metric = GEval(
        name="Technical Correctness",
        criteria="判断 actual_output 相对于 expected_output 是否技术正确。",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=geval_judge,
    )

    grounded = BinaryJudgementNode(
        criteria="Is every factual claim in the actual output supported by retrieval context?",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        label="Groundedness",
    )
    grounded.add_verdict(verdict=False, score=3)
    grounded.add_verdict(verdict=True, score=10)

    task_complete = BinaryJudgementNode(
        criteria="Does the actual output answer the user's question?",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        label="Task completion",
    )
    task_complete.add_verdict(verdict=False, score=0)
    task_complete.add_verdict(verdict=True, then=grounded)

    dag_metric = DAGMetric(
        name="Composite Quality",
        dag=DeepAcyclicGraph(root_nodes=[task_complete]),
        threshold=0.8,
        model=standard_judge,
    )
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
        retrieval_context=["Paris is the capital of France."],
    )
    metrics = [
        AnswerRelevancyMetric(threshold=0.8, model=standard_judge),
        FaithfulnessMetric(threshold=0.8, model=standard_judge),
        correctness_metric,
        dag_metric,
    ]
    assert_test(test_case, metrics)
    return metrics


if __name__ == "__main__":
    run_deepeval_dag_geval()
    print("OK")
