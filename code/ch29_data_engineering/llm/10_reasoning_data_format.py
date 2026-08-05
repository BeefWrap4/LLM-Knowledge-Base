# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: data_engineering.reasoning_data_format
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 10_reasoning_data_format.py
# expected_runtime: <1s
# expected_output: 推理样本的统计信息与元数据
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
#
# Interview hooks:
#   1. R1-Distill 与 s1 长 CoT 数据在"Think" 段的设计上有何本质区别？
#   2. "Budget Forcing" 为什么需要推理数据 think 段支持"可截断、可延长"？
#   3. 如何用 Process Reward Model 验证推理数据的中间步骤正确性？

import json

# 推理数据格式示例（R1 / s1 风格）
reasoning_data_sample = {
    "question": "证明：任意 n 个正整数中，必存在若干个数，其和能被 n 整除。",
    "think": """让我思考一下这个问题。

设这 n 个数为 a_1, a_2, ..., a_n。

考虑前缀和 S_0=0, S_1=a_1, S_2=a_1+a_2, ..., S_n=a_1+...+a_n。

总共有 n+1 个前缀和。

考虑它们模 n 的余数，可能的余数只有 0, 1, ..., n-1 共 n 种。

由鸽笼原理，n+1 个前缀和中必有两个模 n 同余，设为 S_i 和 S_j（i<j）。

那么 S_j - S_i = a_{i+1} + a_{i+2} + ... + a_j 能被 n 整除。

等等，我需要验证一下边界情况。如果 i=0，则 S_j 本身就能被 n 整除，也符合条件。

所以证明成立。让我整理一下完整的证明过程。""",
    "answer": """**证明**：设这 n 个正整数为 a_1, a_2, ..., a_n。

构造前缀和：S_0=0, S_k=a_1+a_2+...+a_k（1≤k≤n）。

共有 n+1 个前缀和（包括 S_0=0），它们模 n 的余数只有 n 种可能（0, 1, ..., n-1）。

**由鸽笼原理**，必有两个前缀和 S_i, S_j（0≤i<j≤n）模 n 同余。

则 S_j - S_i = a_{i+1} + a_{i+2} + ... + a_j ≡ 0 (mod n)。

即存在若干个数（a_{i+1} 到 a_j），其和能被 n 整除。■""",
    "metadata": {
        "domain": "math_olympiad",
        "difficulty": "hard",
        "think_tokens": 287,
        "answer_tokens": 195,
        "source": "s1_v1",
        "verified": True,
    },
}


def main():
    # 统计与一致性校验
    meta = reasoning_data_sample["metadata"]
    think_text = reasoning_data_sample["think"]
    answer_text = reasoning_data_sample["answer"]

    # 简单粗略的 token 估算（真实场景应使用具体 tokenizer）
    estimated_think_tokens = len(think_text)
    estimated_answer_tokens = len(answer_text)
    print(f"题目长度: {len(reasoning_data_sample['question'])} chars")
    print(f"Think 段长度: {estimated_think_tokens} chars (元数据标注 {meta['think_tokens']} tokens)")
    print(f"Answer 段长度: {estimated_answer_tokens} chars (元数据标注 {meta['answer_tokens']} tokens)")
    print(f"领域: {meta['domain']}, 难度: {meta['difficulty']}, 已验证: {meta['verified']}")

    # 校验关键质量指标
    assert meta["verified"], "推理数据必须经过正确性验证"
    assert meta["think_tokens"] >= 100, "Think 段太短，可能不是真正的 CoT"
    assert "答案" not in think_text or "答案" in answer_text, "Think 段不应直接泄露答案"
    print("\n质量校验通过")

    print("\n完整推理样本 JSON（前 200 字符）:")
    print(json.dumps(reasoning_data_sample, ensure_ascii=False)[:200] + "...")
    print("OK")


if __name__ == "__main__":
    main()
