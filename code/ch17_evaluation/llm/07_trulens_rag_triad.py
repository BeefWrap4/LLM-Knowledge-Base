# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.3 TruLens RAG 三元组
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: trulens_eval
# run: python 07_trulens_rag_triad.py
# expected_runtime: <2s (mock mode)
# expected_output: Confirmation that RAG Triad feedback functions are defined
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What is the "RAG Triad" and why is it powerful?
# - How does Groundedness differ from Faithfulness?
# - Why does TruLens emphasize span-level instrumentation?

"""TruLens RAG 三元组评估示例。

TruLens 提出了 RAG Triad 评估框架：
- Answer Relevance：回答是否回答了问题？
- Context Relevance：检索的上下文是否与问题相关？
- Groundedness：回答是否基于检索的上下文（无幻觉）？
"""

import os


def setup_trulens_rag_triad() -> None:
    mock_mode = os.environ.get("TRULENS_MOCK", "1") == "1"

    if mock_mode:
        print("[mock] TruLens RAG Triad 配置示例")
        print("Feedback functions defined:")
        print("  - Answer Relevance: 0.0-1.0")
        print("  - Context Relevance: 0.0-1.0")
        print("  - Groundedness: 0.0-1.0")
        return

    try:
        from trulens_eval import Select, Tru
        from trulens_eval.feedback import Feedback
        from trulens_eval.feedback.provider.openai import OpenAI as OpenAIFeedback
    except ImportError as exc:
        print(f"[mock] trulens_eval 未安装 ({exc})，使用模拟输出")
        return

    # 初始化
    tru = Tru()
    provider = OpenAIFeedback()

    # 定义三元组 Feedback 函数
    f_answer_relevance = Feedback(
        provider.relevance,
        name="Answer Relevance",
    ).on_input_output()

    f_context_relevance = (
        Feedback(provider.context_relevance, name="Context Relevance")
        .on_input()
        .on(Select.RecordCalls.retrieve.rets[:])
    )

    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
        )
        .on(Select.RecordCalls.retrieve.rets[:])
        .on_output()
    )

    print("RAG Triad 已定义:")
    print(f"  {f_answer_relevance.name}")
    print(f"  {f_context_relevance.name}")
    print(f"  {f_groundedness.name}")

    # 使用 TruChain 包装 RAG 链进行评估
    # tru_chain = TruChain(rag_chain, feedbacks=[f_answer_relevance, ...])
    # with tru_chain as recording:
    #     response = rag_chain.invoke("用户问题")
    # records, feedback = tru.get_records_and_feedback(...)


if __name__ == "__main__":
    setup_trulens_rag_triad()
