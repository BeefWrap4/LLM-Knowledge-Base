# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.11.4 DeepEval DAG + G-Eval
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: deepeval, pytest
# run: python 14_deepeval_dag_geval.py
# expected_runtime: <2s (mock mode) / 30-60s (real)
# expected_output: Eval-as-Code DAG + G-Eval demonstration
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What is DAG-based evaluation in DeepEval 2025+ and why is it useful?
# - How does G-Eval let domain experts define metrics without code?
# - How do you run DeepEval in CI: `deepeval test run` and what does it produce?

"""DeepEval DAG + G-Eval + Pytest 风格示例。

DeepEval 在 2025-2026 完成了重大架构升级：
引入 DAG-based evaluation（基于有向无环图的多步评估）和 G-Eval 框架。
"""

import os


def run_deepeval_dag_geval() -> None:
    mock_mode = os.environ.get("DEEPEVAL_MOCK", "1") == "1"

    if mock_mode:
        print("[mock] DeepEval DAG + G-Eval 演示")
        print("[mock] G-Eval 'Technical Correctness' (LLM: gpt-4o, threshold=0.7)")
        print("[mock] DAG 节点:")
        print("[mock]   - Task Completion (TaskCompletionIndicator)")
        print("[mock]   - No Hallucination (HallucinationIndicator)")
        print("[mock] DAGMetric: composite score 0.86 (threshold=0.8)")
        print("[mock] 运行命令: deepeval test run test_rag.py")
        return

    try:
        from deepeval import assert_test
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            DAGMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.metrics.dag import DeepAcyclicGraph
        from deepeval.metrics.indicators import (
            HallucinationIndicator,
            TaskCompletionIndicator,
        )
        from deepeval.models import GPTModel
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as exc:
        print(f"[mock] deepeval 未安装 ({exc})，使用模拟输出")
        return

    # 1. G-Eval：自然语言定义 metric（领域专家可写）
    correctness_metric = GEval(
        name="Technical Correctness",
        criteria="判断 actual_output 相对于 expected_output 是否技术正确，评估对象是高级 Python 开发者。",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=GPTModel(model="gpt-4o"),
    )

    # 2. DAG-based 多步评估
    dag = DeepAcyclicGraph()

    task_complete = TaskCompletionIndicator(
        name="Task Completion",
        criteria="Whether the response actually addresses the user's question.",
    )
    no_hallucination = HallucinationIndicator(
        name="No Hallucination",
        criteria="Whether the output contains information not present in the context.",
    )

    # 组合 DAG：先检查任务完成 → 再检查幻觉
    dag.add_node("Task Completion", task_complete)
    dag.add_node(
        "No Hallucination",
        no_hallucination,
        dependencies=["Task Completion"],
    )

    dag_metric = DAGMetric(name="Composite Quality", dag=dag, threshold=0.8)

    # 3. Pytest 风格测试
    def test_rag_pipeline_quality():
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris, population 2.1 million.",
            expected_output="Paris",
            retrieval_context=["Paris is the capital and most populous city of France."],
        )
        assert_test(
            test_case,
            [
                AnswerRelevancyMetric(threshold=0.8),
                FaithfulnessMetric(threshold=0.8),
                correctness_metric,
                dag_metric,
            ],
        )

    # 运行：deepeval test run test_rag.py
    # 失败时自动给出失败原因、链接到 Confident AI Dashboard
    test_rag_pipeline_quality()


if __name__ == "__main__":
    run_deepeval_dag_geval()
