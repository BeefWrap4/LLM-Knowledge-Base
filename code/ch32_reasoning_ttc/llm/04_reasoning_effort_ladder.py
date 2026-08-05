# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.reasoning_effort_ladder
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0, DEEPSEEK_API_KEY
# run: python 04_reasoning_effort_ladder.py
# expected_runtime: <90s (real DeepSeek V4 Pro reasoning)
# expected_output: compares V4 Pro high/max thinking effort + final answer
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. DeepSeek V4 的 high/max 与 OpenAI 模型的 reasoning_effort 档位有何差异？
#   2. reasoning_content 字段与 final content 分离的设计意义？
#   3. 为什么不能把 reasoning_effort 直接换算成固定 token 数或质量增益？
"""Reasoning Effort 阶梯 (DeepSeek V4 Pro).

推理模型: 增加 inference-time compute 提升质量
  - high: 官方常规请求默认档位
  - max:  更高思考强度，具体成本与收益需按任务实测

DeepSeek V4 兼容映射会把 low/medium 映射为 high，把 xhigh 映射为 max；
教学示例直接使用官方原生的 high/max，避免把映射档位误当成四种能力。
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
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("=== DeepSeek V4 Pro reasoning_effort（离线演示）===")
        for effort in ("high", "max"):
            print(
                f"reasoning_effort={effort}, "
                "thinking={'type': 'enabled'}, model=deepseek-v4-pro"
            )
        return

    api_key = get_deepseek_key()

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    print("=== DeepSeek V4 Pro 推理 effort 阶梯 ===\n")

    question = "9.11 和 9.9 哪个更大? 详细推理过程"

    for effort in ("high", "max"):
        print(f"\n--- reasoning_effort={effort} ---")
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": question}],
            max_tokens=2048,
            reasoning_effort=effort,
            extra_body={"thinking": {"type": "enabled"}},
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", "") or ""
        content = msg.content or ""
        print(f"推理 (前 300 字符): {reasoning[:300]}")
        print(f"\n最终回答: {content[:300]}")
        if resp.usage:
            print(f"\nusage: {resp.usage.total_tokens} tokens (含 reasoning)")


if __name__ == "__main__":
    main()
    print("OK")
