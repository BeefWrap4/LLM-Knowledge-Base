# ---
# chapter: 27
# topic: Claude 4.6 Extended Thinking + Interleaved Thinking
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: anthropic>=0.40.0
# run: python 03_claude_extended_thinking.py
# expected_runtime: <60s (real Anthropic API call with thinking blocks)
# expected_output: prints <thinking>...</thinking> block + final answer, or friendly error
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3
# Interview hooks:
#   1. Claude Extended Thinking 与 o3 reasoning_effort 的本质区别？
#   2. thinking budget_tokens 与 max_tokens 的关系？interleaved thinking？
#   3. 如何在 tool_use 中启用 extended thinking？
"""Claude Extended Thinking (Anthropic).

Extended thinking 让 Claude 在生成最终回答前显式思考:
  - 思考内容: <thinking>...</thinking> 块 (用户可见)
  - 最大思考 token: thinking budget (max_tokens 参数)
  - 适合: 复杂推理, 多步问题
"""

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help


def get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY":
        raise_with_help(
            "ANTHROPIC_API_KEY 未设置",
            "国内用户可用 DeepSeek-R1 替代.",
        )
    return key


def main():
    api_key = get_anthropic_key()

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    print("=== Claude Extended Thinking ===\n")

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        thinking={
            "type": "enabled",
            "budget_tokens": 2048,
        },
        messages=[{"role": "user", "content": "9.11 和 9.9 哪个更大?"}],
    )

    for block in response.content:
        if block.type == "thinking":
            print(f"\n<thinking>\n{block.thinking[:500]}\n</thinking>")
        elif block.type == "text":
            print(f"\n回答: {block.text}")


if __name__ == "__main__":
    main()
