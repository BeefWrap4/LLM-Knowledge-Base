# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.claude_extended_thinking
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: anthropic>=0.120.2,<1
# run: python 03_claude_extended_thinking.py
# expected_runtime: <2s mock; variable for a real Anthropic API call
# expected_output: prints a summarized thinking block and final answer
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. Claude Fable 5 为什么不能继续使用 ``budget_tokens``？
#   2. ``thinking.display`` 的 summarized 与 omitted 分别返回什么？
#   3. ``output_config.effort`` 为什么不是严格的 token 预算？
"""Claude Fable 5 的 always-on adaptive thinking 示例。

Fable 5 不接受旧式 ``thinking.type="enabled"`` / ``budget_tokens``。推理深度由
``output_config.effort`` 控制；``thinking.display="summarized"`` 只返回可读摘要，
不会返回原始思维链。脚本默认离线 mock，只有 ``LLM_MOCK=0`` 才调用真实 API。
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
            "真实调用需同时设置 `LLM_MOCK=0` 与 `ANTHROPIC_API_KEY`；离线运行请保留默认 mock。",
        )
    return key


def main():
    if os.environ.get("LLM_MOCK", "1").strip() != "0":
        print("=== Claude Fable 5 adaptive thinking（离线 mock，默认）===")
        print("<thinking-summary>这是可读摘要示例，不是原始思维链。</thinking-summary>")
        print("回答: 9.9 更大。")
        return

    api_key = get_anthropic_key()

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    print("=== Claude Fable 5 adaptive thinking ===\n")

    response = client.messages.create(
        model="claude-fable-5",
        max_tokens=4096,
        thinking={
            "type": "adaptive",
            "display": "summarized",
        },
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": "9.11 和 9.9 哪个更大?"}],
    )

    for block in response.content:
        if block.type == "thinking":
            summary = block.thinking or "(未返回可读摘要)"
            print(f"\n<thinking-summary>\n{summary[:500]}\n</thinking-summary>")
        elif block.type == "text":
            print(f"\n回答: {block.text}")


if __name__ == "__main__":
    main()
    print("OK")
