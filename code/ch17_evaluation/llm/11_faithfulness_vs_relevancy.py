# ---
# chapter: 17
# topic: 大模型评估体系
# section: 高频题 - Faithfulness vs Answer Relevancy
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (none, illustrative)
# run: python 11_faithfulness_vs_relevancy.py
# expected_runtime: <1s
# expected_output: Side-by-side examples illustrating each combination
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Give an example of high Faithfulness + low Answer Relevancy.
# - Give an example of low Faithfulness + high Answer Relevancy.
# - Why is the "ideal" state high on both axes?

"""Faithfulness vs Answer Relevancy 概念说明示例。

通过具体例子展示 RAG 评估中两个核心指标的差异。
"""


def main() -> None:
    context = "Python 3.13 引入了新的 GIL 实现"
    question = "Python 3.13 有哪些新特性？"

    print(f"上下文: {context}")
    print(f"问题: {question}")
    print()

    # 高 Faithfulness + 低 Relevancy:
    answer_a = "Python 3.13 引入了新的 GIL 实现（仅此而已）"
    print("[案例 A] 高 Faithfulness + 低 Relevancy（忠实但不全面）:")
    print(f"  回答: {answer_a}")
    print()

    # 低 Faithfulness + 高 Relevancy:
    answer_b = "Python 3.13 引入了新的 GIL 实现和 JIT 编译器和模式匹配"
    print("[案例 B] 低 Faithfulness + 高 Relevancy（可能相关但 JIT 和模式匹配非真 = 幻觉）:")
    print(f"  回答: {answer_b}")
    print()

    # 高 Faithfulness + 高 Relevancy: 是最理想的情况
    answer_ideal = (
        "Python 3.13 引入了新的实验性 GIL 实现（PEP 703），"
        "在保证现有兼容性的同时探索无 GIL 模式的可能。"
    )
    print("[案例 理想] 高 Faithfulness + 高 Relevancy（最理想）:")
    print(f"  回答: {answer_ideal}")
    print()

    print("结论:")
    print("- Faithfulness 检查 '是否编造'（回答 ⊆ 上下文）")
    print("- Answer Relevancy 检查 '是否切题'（回答 ↔ 问题）")
    print("- 两个指标互相独立、缺一不可")


if __name__ == "__main__":
    main()
    print("OK")
