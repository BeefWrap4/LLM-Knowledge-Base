# ---
# chapter: 27
# topic: Reasoning Effort 决策阶梯 + 成本/延迟预算
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0, DEEPSEEK_API_KEY
# run: python 04_reasoning_effort_ladder.py
# expected_runtime: <90s (real DeepSeek R1 reasoning)
# expected_output: prints reasoning chain (R1 chain-of-thought) + final answer
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2
# Interview hooks:
#   1. DeepSeek-R1 与 o3 的 reasoning_effort 表达方式区别？R1 是隐式还是显式？
#   2. reasoning_content 字段与 final content 分离的设计意义？
#   3. R1 的 chain-of-thought 平均长度？是否对 max_tokens 有下限要求？
"""Reasoning Effort 阶梯 (DeepSeek-R1 + OpenAI o3 对比).

推理模型: 增加 inference-time compute 提升质量
  - low:    短推理, 快速回答
  - medium: 平衡
  - high:   长推理, 慢但准
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
            "DEEPSEEK_API_KEY 未设置",
            "运行 `make llm-doctor-setup` 配置.",
        )
    return key


def main():
    api_key = get_deepseek_key()

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )

    print("=== DeepSeek-R1 推理 effort 阶梯 ===\n")

    question = "9.11 和 9.9 哪个更大? 详细推理过程"

    for effort_desc, model in [("R1 默认 (deepseek-reasoner)", "deepseek-reasoner")]:
        print(f"\n--- {effort_desc} ---")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", "") or ""
        content = msg.content or ""
        print(f"推理 (前 300 字符): {reasoning[:300]}")
        print(f"\n最终回答: {content[:300]}")
        print(f"\nusage: {resp.usage.total_tokens} tokens (含 reasoning)")


if __name__ == "__main__":
    main()
