# ---
# chapter: 27
# topic: o3 streaming with hidden reasoning tokens
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0
# run: python 02_o3_streaming.py
# expected_runtime: <60s (real streaming API call)
# expected_output: streams code blocks live, or friendly error if no key
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2
# Interview hooks:
#   1. 流式响应的 delta 字段结构？reasoning_content 与 content 分流？
#   2. 为什么流式推理比 batch 推理的 TTFT (time-to-first-token) 短？
#   3. o3 streaming 与 o1 的 SDK 兼容性差异？
"""OpenAI o3 流式推理 (真实 streaming)."""

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
            "国内用户可用 DeepSeek-R1 替代: `export DEEPSEEK_API_KEY=...` + 改 base_url.",
        )
    return key


def main():
    api_key = get_openai_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    print("=== OpenAI o3 流式推理 ===\n")

    stream = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": "写一个 Python 函数判断回文"}],
        reasoning_effort="medium",
        max_completion_tokens=2048,
        stream=True,
    )

    full_text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_text += delta
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
