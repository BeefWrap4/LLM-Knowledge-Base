# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.2 Ragas 框架
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: ragas, datasets, langchain-openai
# run: python 06_ragas_evaluation.py
# expected_runtime: <5s (mock mode) / 30-60s (real)
# expected_output: RAG evaluation scores for faithfulness, relevancy, recall, precision
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Explain the difference between Faithfulness and Answer Relevancy.
# - How does Ragas compute Context Recall without ground-truth labels?
# - What are the limitations of LLM-as-Judge based RAG metrics?

"""Ragas 实战评估示例。

Ragas（RAG Assessment）是最流行的开源 RAG 评估框架，
提供忠实度、答案相关性、上下文召回率、上下文精确率等核心指标。
"""
import os


def run_ragas_evaluation() -> None:
    mock_mode = os.environ.get("RAGAS_MOCK", "1") == "1"

    if mock_mode:
        print("[mock] 模拟 Ragas 评估输出（设置 RAGAS_MOCK=0 走真实评估）")
        print("RAG 评估结果：")
        print("  faithfulness      answer_relevancy  context_recall  context_precision")
        print("0          0.95              0.92             0.88               0.91")
        print("1          0.93              0.94             0.90               0.89")
        return

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        print(f"[mock] 依赖未安装 ({exc})，使用模拟输出")
        return

    # 准备评估数据
    eval_dataset = Dataset.from_dict(
        {
            "question": [
                "什么是Transformer的注意力机制？",
                "Python中的GIL是什么？",
            ],
            "answer": [
                "注意力机制是Transformer的核心组件，它允许模型在处理序列时动态关注不同位置的信息。"
                "通过计算Query、Key、Value之间的相似度，模型可以为不同位置的token分配不同的权重，"
                "从而捕捉长距离依赖关系。",
                "GIL（全局解释器锁）是CPython中的一种互斥锁，它确保同一时刻只有一个线程执行Python字节码。"
                "这简化了CPython的内存管理，但也限制了多线程程序的并行性能。",
            ],
            "contexts": [
                [
                    "注意力机制通过Q、K、V三个矩阵计算token间的关系。",
                    "Transformer架构由Vaswani等人在2017年提出，核心创新是自注意力机制。",
                    "自注意力可以捕捉序列中任意两个位置之间的依赖关系。",
                ],
                [
                    "GIL是CPython解释器中的全局锁，用于保护内部数据结构。",
                    "由于GIL的存在，多线程Python程序在CPU密集型任务中无法充分利用多核。",
                    "可以通过multiprocessing模块或使用PyPy/Jython等替代解释器绕过GIL。",
                ],
            ],
            "ground_truth": [
                "注意力机制是Transformer中让模型关注输入序列中不同部分重要性的机制，通过Q、K、V计算实现。",
                "GIL是CPython的全局解释器锁，确保线程安全但限制了多线程并行。",
            ],
        }
    )

    # 使用 Ragas 评估（需要配置 OpenAI Key 用于基于 LLM 的指标）
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

    # 选择评估指标
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    # 执行评估
    result = evaluate(dataset=eval_dataset, metrics=metrics, llm=llm)

    # 输出结果
    df = result.to_pandas()
    print("RAG 评估结果：")
    print(
        df[
            [
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "context_precision",
            ]
        ]
    )


if __name__ == "__main__":
    run_ragas_evaluation()
