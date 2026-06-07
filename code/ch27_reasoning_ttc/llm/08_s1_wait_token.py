# ---
# chapter: 27
# topic: s1 Wait token extension + budget extrapolation
# section: 27.3.3 s1
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0, DEEPSEEK_API_KEY
# run: python 08_s1_wait_token.py
# expected_runtime: <90s (real 2-round DeepSeek R1 with wait trigger)
# expected_output: prints 2-round comparison (no-wait vs with-wait)
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3.3
# Interview hooks:
#   1. "Wait" 比 "Therefore" 触发继续思考效果更好的原因？
#   2. wait token 的位置 (开头/中间/结尾) 对续写长度的影响？
#   3. 如何把 wait token 蒸馏到小模型，避免推理时外部控制？
"""S1 Wait Token 策略.

Wait Token: 在 reasoning 末尾追加 "Wait" 触发模型继续思考.
S1 论文发现, 简单 "Wait" 比无触发平均提升 30% 准确率.
"""
import sys
import os
from pathlib import Path
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help


def get_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY":
        raise_with_help("DEEPSEEK_API_KEY 未设置", "运行 `make llm-doctor-setup`.")
    return key


def wait_token_demo():
    api_key = get_deepseek_key()

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    print("=== S1 Wait Token 演示 ===\n")

    question = "Solve: If x + 2y = 5 and 3x - y = 1, find x and y."

    # 第一轮: 不加 wait
    print("--- 轮 1: 不加 wait ---")
    resp1 = client.chat.completions.create(
        model="deepseek-reasoner", messages=[{"role": "user", "content": question}],
        max_tokens=2048,
    )
    msg1 = resp1.choices[0].message
    reasoning1 = getattr(msg1, "reasoning_content", "") or ""
    content1 = msg1.content or ""
    print(f"  reasoning: {len(reasoning1)} chars")
    print(f"  answer: {content1[:200]}")

    # 第二轮: 加 "Wait" 继续
    if not content1 or "?" in content1:
        print(f"\n--- 轮 2: 加 'Wait' 继续 ---")
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reasoning1 + content1},
            {"role": "user", "content": "Wait, verify your answer and explain more carefully."},
        ]
        resp2 = client.chat.completions.create(
            model="deepseek-reasoner", messages=messages, max_tokens=2048,
        )
        msg2 = resp2.choices[0].message
        reasoning2 = getattr(msg2, "reasoning_content", "") or ""
        content2 = msg2.content or ""
        print(f"  reasoning: {len(reasoning2)} chars (额外)")
        print(f"  answer: {content2[:200]}")


def main():
    wait_token_demo()


if __name__ == "__main__":
    main()
