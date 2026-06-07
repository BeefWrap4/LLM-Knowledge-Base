# ---
# chapter: 27
# topic: OpenAI o3 API basic usage (real OpenAI API)
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0
# run: python 01_o3_api_basic.py
# expected_runtime: <60s (real API call, 3 reasoning_effort levels)
# expected_output: prints 3 reasoning_effort responses or friendly error if no key
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2
# Interview hooks:
#   1. o3 / o4-mini 的 reasoning_effort 参数含义？三档对应什么？
#   2. 为什么 o-series 不能用 temperature=0 的常见做法？默认采样超参是什么？
#   3. reasoning_effort="high" 与 max_completion_tokens 的关系？
"""OpenAI o3 推理模型 (真实 OpenAI API).

o3 是 OpenAI 的 reasoning model, 通过 reasoning_effort 控制推理深度:
  - low: 快速回答
  - medium: 平衡
  - high: 深度推理 (类比 o1-pro)
"""
import sys
import os
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
            "OpenAI o3 需付费 API. 国内用户可用 `make llm-doctor-setup` 配置 DeepSeek 替代.",
        )
    return key


def main():
    api_key = get_openai_key()

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print("=== OpenAI o3 推理模型 ===\n")

    # reasoning_effort 对比
    question = "9.11 和 9.9 哪个更大? 详细推理"

    for effort in ["low", "medium", "high"]:
        try:
            resp = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": question}],
                reasoning_effort=effort,
                max_completion_tokens=1024,
            )
            content = resp.choices[0].message.content
            print(f"\n--- reasoning_effort={effort} ---")
            print(content[:500] if content else "(空, 仅 reasoning_tokens)")
            print(f"usage: {resp.usage.total_tokens} total")
        except Exception as e:
            print(f"\n--- reasoning_effort={effort} ---")
            print(f"❌ {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
