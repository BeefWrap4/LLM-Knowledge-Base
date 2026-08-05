# ---
# chapter: 37
# topic: RAG、Agent 与安全评估
# topic_id: evaluation.faithfulness_vs_relevancy
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (none, illustrative)
# run: python 11_faithfulness_vs_relevancy.py
# expected_runtime: <1s
# expected_output: Side-by-side examples illustrating each combination
# ---
# See: ../../../37_RAG_Agent与安全评估.md
# Interview hooks:
# - Give an example of high Faithfulness + low Answer Relevancy.
# - Give an example of low Faithfulness + high Answer Relevancy.
# - Why is the "ideal" state high on both axes?

"""Faithfulness vs Answer Relevancy 概念说明示例。

通过具体例子展示 RAG 评估中两个核心指标的差异。
"""


def main() -> None:
    context = "退款政策：购买后 30 天内可凭原始收据退款；客服时间为工作日 9:00-18:00。"
    question = "我购买商品 20 天了，应该怎样退款？"

    print(f"上下文: {context}")
    print(f"问题: {question}")
    print()

    # 高 Faithfulness + 低 Relevancy:
    answer_a = "客服时间为工作日 9:00-18:00。"
    print("[案例 A] 高 Faithfulness + 低 Relevancy（有依据但没有回答退款条件）:")
    print(f"  回答: {answer_a}")
    print()

    # 低 Faithfulness + 高 Relevancy:
    answer_b = "购买后 60 天内都可以无收据退款。"
    print("[案例 B] 低 Faithfulness + 高 Relevancy（切题但与上下文冲突）:")
    print(f"  回答: {answer_b}")
    print()

    # 高 Faithfulness + 高 Relevancy: 是最理想的情况
    answer_ideal = "你仍在 30 天期限内，请携带原始收据申请退款。"
    print("[案例 理想] 高 Faithfulness + 高 Relevancy（最理想）:")
    print(f"  回答: {answer_ideal}")
    print()

    print("结论:")
    print("- Faithfulness 检查 '是否编造'（回答 ⊆ 上下文）")
    print("- Answer Relevancy 检查 '是否切题'（回答 ↔ 问题）")
    print("- 两者衡量不同维度，实测可能相关；都需结合业务验收标准")


if __name__ == "__main__":
    main()
    print("OK")
