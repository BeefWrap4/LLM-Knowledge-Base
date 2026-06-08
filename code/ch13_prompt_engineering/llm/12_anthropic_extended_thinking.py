# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.1 Anthropic Extended Thinking
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic
# run: python 12_anthropic_extended_thinking.py
# expected_runtime: 10-30s (real api)
# expected_output: 打印 thinking block 与 final answer
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.1
# Interview hooks:
# - Extended Thinking 与普通 CoT Prompt 的本质区别？
# - budget_tokens 太大太小各有什么风险？
# - thinking tokens 是否计费？为什么需要单独区分？

import anthropic

_client = anthropic.Anthropic()


def call_anthropic(question: str):
    return _client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 5000,  # 最多使用 5000 tokens 进行思考
        },
        messages=[{"role": "user", "content": question}],
    )


if __name__ == "__main__":
    question = (
        "一家公司有 3 个仓库，分别有 2400/1800/3000 件商品。"
        "需按 7:3 比例分配到区域 A 和 B。仓库 A 到 A/B 距离 10/25km，"
        "仓库 B 到 A/B 距离 15/10km，仓库 C 到 A/B 距离 20/5km，"
        "单位运输成本 2 元/km/件。求最小化运输成本的分配方案。"
    )

    response = call_anthropic(question)

    # 响应包含两个 block：thinking block 和 text block
    for block in response.content:
        if block.type == "thinking":
            print(f"【思考过程】{block.thinking[:500]}...")
        elif block.type == "text":
            print(f"【最终答案】{block.text}")

    print(f"输入 tokens:  {response.usage.input_tokens}")
    print(f"输出 tokens:  {response.usage.output_tokens}")
    print(f"思考 tokens:  {getattr(response.usage, 'thinking_tokens', 'N/A')}")
