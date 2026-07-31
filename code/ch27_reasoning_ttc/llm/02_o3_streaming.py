# ---
# chapter: 27
# topic: GPT-5.6 Sol streaming with the Responses API
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=2.51.0,<3
# run: python 02_o3_streaming.py
# expected_runtime: <2s mock; variable for a real streaming API call
# expected_output: streams text deltas; real API requires LLM_MOCK=0 and a key
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2
# Interview hooks:
#   1. Responses API 的 ``response.output_text.delta`` 事件如何消费？
#   2. 流式传输为何改善感知延迟，却不保证降低服务端首个文本 token 的生成时间？
#   3. 为什么不应把未返回的原始推理链当作可流式读取字段？
"""用 GPT-5.6 Sol + Responses API 演示文本增量流。

文件名保留 ``o3`` 仅为兼容既有教程链接。脚本默认离线 mock；只有显式设置
``LLM_MOCK=0`` 才会调用真实 API。Responses API 返回可展示的文本增量，而不是原始思维链。
"""

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY":
        raise_with_help(
            "OPENAI_API_KEY 未设置",
            "真实调用需同时设置 `LLM_MOCK=0` 与 `OPENAI_API_KEY`；离线运行请保留默认 mock。",
        )
    return key


def main():
    if os.environ.get("LLM_MOCK", "1").strip() != "0":
        print("=== GPT-5.6 Sol Responses 流（离线 mock，默认）===")
        for delta in ["def ", "is_palindrome", "(text): ", "return text == text[::-1]"]:
            print(delta, end="", flush=True)
        print()
        return

    api_key = get_openai_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    print("=== GPT-5.6 Sol + Responses API 流式文本 ===\n")

    stream = client.responses.create(
        model="gpt-5.6-sol",
        input="写一个 Python 函数判断回文，并给出两个测试样例。",
        reasoning={"effort": "medium"},
        max_output_tokens=2048,
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
    print("OK")
