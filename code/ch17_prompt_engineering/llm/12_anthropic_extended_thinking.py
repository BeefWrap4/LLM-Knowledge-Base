# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: prompt_engineering.anthropic_extended_thinking
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic
# run: python 12_anthropic_extended_thinking.py
# expected_runtime: 10-30s (real api)
# expected_output: 打印 thinking block 与 final answer
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
# - Extended Thinking 与普通 CoT Prompt 的本质区别？
# - adaptive thinking 与旧版 budget_tokens 的迁移差异？
# - 为什么应通过评测选择 effort，而不是默认最高档？

import os

try:
    import anthropic
except ImportError:
    anthropic = None


def _real_api_ready() -> bool:
    if os.environ.get("LLM_MOCK") != "0":
        print("[SKIP] 离线安全模式：只有显式设置 LLM_MOCK=0 才会调用 Anthropic API")
        print("OK")
        return False
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        print("[SKIP] 真实调用需要 anthropic 和 ANTHROPIC_API_KEY")
        print("OK")
        return False
    return True


def call_anthropic(question: str):
    if not _real_api_ready():
        return None
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
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
    if response is None:
        raise SystemExit(0)

    # thinking block 是否出现由模型和响应决定；不要假设固定有两个 block
    for block in response.content:
        if block.type == "thinking":
            print(f"【API 提供的 thinking 内容】{block.thinking[:500]}...")
        elif block.type == "text":
            print(f"【最终答案】{block.text}")

    print(f"输入 tokens:  {response.usage.input_tokens}")
    print(f"输出 tokens:  {response.usage.output_tokens}")
    print("OK")
