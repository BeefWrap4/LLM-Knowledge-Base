# ---
# chapter: 27
# topic: Claude 4.6 Extended Thinking + Interleaved Thinking
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: anthropic>=0.40.0 (optional)
# run: python 03_claude_extended_thinking.py
# expected_runtime: <2s
# expected_output: prints mock extended thinking blocks
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2 + §27.8 Q7
# Interview hooks:
#   1. Claude Extended Thinking 的 budget_tokens 和 max_tokens 区别？
#   2. 什么是 Interleaved Thinking？为何对 Agent 关键？
#   3. thinking block 在多轮对话中如何回传？
"""Claude 4.6 Extended Thinking：budget_tokens 控制思考预算。

核心 API:
  thinking = {"type": "enabled", "budget_tokens": 5000}
  返回中包含 type="thinking" 的 block（可选择重加密回传），
  以及最终的 type="text" answer block。
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class ThinkingBlock:
    type: str  # "thinking" | "text" | "tool_use"
    content: str
    tokens: int


def _mock_extended_thinking(
    question: str, budget_tokens: int
) -> list[ThinkingBlock]:
    """Mock 思考过程：按 budget 比例生成 thinking 与 text。"""
    random.seed(hash(question) % 999)
    used = random.randint(int(budget_tokens * 0.7), budget_tokens)
    think = (
        "The user asks about a non-trivial problem. Let me decompose:\n"
        "  1. Identify known facts...\n"
        "  2. Consider edge cases...\n"
        "  3. Verify by substitution...\n"
    ) * (used // 100 + 1)
    think = think[: used * 4]  # 粗略 4 字符/token
    answer = "After careful consideration, the answer is 42."
    return [
        ThinkingBlock("thinking", think, used),
        ThinkingBlock("text", answer, len(answer.split())),
    ]


def call_claude(
    question: str, budget_tokens: int = 5000
) -> list[ThinkingBlock]:
    """调用 Claude；缺 API key 时走 mock。"""
    if budget_tokens < 1024:
        raise ValueError("budget_tokens 必须 >= 1024")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=budget_tokens + 4096,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                messages=[{"role": "user", "content": question}],
            )
            blocks = []
            for b in resp.content:
                blocks.append(
                    ThinkingBlock(b.type, getattr(b, "thinking", b.text), 0)
                )
            return blocks
        except Exception as e:  # noqa: BLE001
            print(f"[warn] anthropic 调用失败 ({e})，回退到 mock")
    return _mock_extended_thinking(question, budget_tokens)


def main() -> None:
    q = "Design a function that returns the k-th prime in O(n log log n)."

    # 不同 budget 对比
    for budget in (1024, 8000, 32000):
        blocks = call_claude(q, budget_tokens=budget)
        thinking = next((b for b in blocks if b.type == "thinking"), None)
        text = next((b for b in blocks if b.type == "text"), None)
        n_think = thinking.tokens if thinking else 0
        preview = (thinking.content[:60] + "...") if thinking else ""
        print(
            f"[budget={budget:>5}] think_tokens={n_think:>5}  "
            f"answer={text.content if text else ''!r}"
        )
        print(f"           think_preview={preview!r}")

    print("\n提示: Interleaved Thinking 让模型在 tool_use 之间持续思考，")
    print("      这是 Claude 4.6 在 Agent 场景相对 o3 的关键差异。")


if __name__ == "__main__":
    main()
