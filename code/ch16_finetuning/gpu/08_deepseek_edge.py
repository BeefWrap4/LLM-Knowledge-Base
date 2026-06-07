# ---
# chapter: 16
# topic: DeepSeek Edge (真实 DeepSeek API via OpenAI 协议)
# section: 16.6.4
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: openai SDK
# run: export DEEPSEEK_API_KEY=sk-xxx; python 08_deepseek_edge.py
# expected_runtime: 5-30s (取决于网络与 R1 推理时间)
# expected_output: deepseek-chat + deepseek-reasoner 两个真实响应
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.6.4
#
# Interview hooks:
#   1. DeepSeek API 的 OpenAI 兼容性如何实现？BaseURL 重写？
#   2. deepseek-chat (V3) vs deepseek-reasoner (R1) 的能力差异？
#   3. Reasoning model 的 token 消耗模式？R1 平均比 V3 多 5-10x?
"""DeepSeek Edge API 真实调用 (OpenAI 协议).

DeepSeek API 完全兼容 OpenAI 协议:
  base_url = https://api.deepseek.com/v1
  api_key  = DEEPSEEK_API_KEY env

可用 openai SDK 直接连.
"""
import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help


def get_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY":
        raise_with_help(
            "DEEPSEEK_API_KEY 未设置或为占位符",
            "运行 `export DEEPSEEK_API_KEY=sk-xxx` (Linux/Mac) "
            "或 `$env:DEEPSEEK_API_KEY='sk-xxx'` (PowerShell). "
            "或使用 Ollama 本地替代: `ollama pull deepseek-r1:7b`.",
        )
    return key


def main():
    api_key = get_deepseek_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    print("=== DeepSeek Edge API 真实调用 ===\n")

    # 1) deepseek-chat (V3) — 普通对话
    print("[1/2] deepseek-chat (V3):")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "讲个冷笑话, 一句话"}],
        max_tokens=64,
        temperature=0.7,
    )
    content = resp.choices[0].message.content
    print(f"  回答: {content}")
    if resp.usage:
        print(f"  usage: total_tokens={resp.usage.total_tokens} "
              f"(prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens})")

    # 2) deepseek-reasoner (R1) — 推理模型
    print("\n[2/2] deepseek-reasoner (R1):")
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "9.11 和 9.9 哪个大? 请推理."}],
        max_tokens=512,
    )
    content = resp.choices[0].message.content
    # R1 还会返回 reasoning_content
    reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
    print(f"  回答: {content[:200]}")
    if reasoning:
        print(f"  推理过程: {reasoning[:200]}...")
    if resp.usage:
        print(f"  usage: total_tokens={resp.usage.total_tokens} "
              f"(R1 通常比 V3 消耗多 5-10x)")


if __name__ == "__main__":
    main()
