# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.s1_wait_token
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0, DEEPSEEK_API_KEY
# run: python 08_s1_wait_token.py
# expected_runtime: <90s (real 2-round DeepSeek V4 Pro thinking)
# expected_output: prints initial answer vs follow-up verification
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. "Wait" 比 "Therefore" 触发继续思考效果更好的原因？
#   2. API follow-up prompt 与解码流中注入 Wait token 有什么本质差异？
#   3. 如何用本地推理引擎严格控制停止位置与 reasoning token budget？
"""S1 Wait Token 策略与托管 API 的能力边界.

原始 s1 方法在受控解码过程中追加 Wait token。DeepSeek 托管
Chat Completions API 不暴露隐藏思考流的 token 级注入接口，因此本例仅用
新的 user turn 请求复核。这可演示产品层工作流，但不是论文方法的严格复现，
也不预设质量一定提升。
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
        raise_with_help("DEEPSEEK_API_KEY 未设置", "运行 `make llm-doctor-setup`.")
    return key


def wait_token_demo():
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("=== s1 Wait 策略（DeepSeek V4 API 离线演示）===")
        print("model=deepseek-v4-pro, thinking=enabled, reasoning_effort=high")
        print("轮 1: 初始回答；轮 2: 新 user turn 请求复核（非 token 级 Wait 注入）")
        return

    api_key = get_deepseek_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=== s1 Wait 策略（DeepSeek V4 API 近似）===\n")

    question = "Solve: If x + 2y = 5 and 3x - y = 1, find x and y."

    # 第一轮: 初始思考请求
    print("--- 轮 1: 初始回答 ---")
    resp1 = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": question}],
        max_tokens=2048,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    msg1 = resp1.choices[0].message
    reasoning1 = getattr(msg1, "reasoning_content", "") or ""
    content1 = msg1.content or ""
    print(f"  reasoning: {len(reasoning1)} chars")
    print(f"  answer: {content1[:200]}")

    # 第二轮: 新 user turn 请求复核。非工具场景无需回传上一轮隐藏思考。
    print("\n--- 轮 2: follow-up 请求复核 ---")
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": content1 or "(尚未给出最终答案)"},
        {"role": "user", "content": "Wait. Verify the answer and explain any correction carefully."},
    ]
    resp2 = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        max_tokens=2048,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    msg2 = resp2.choices[0].message
    reasoning2 = getattr(msg2, "reasoning_content", "") or ""
    content2 = msg2.content or ""
    print(f"  reasoning: {len(reasoning2)} chars")
    print(f"  answer: {content2[:200]}")
    print("\n  注意: 两轮字符数仅是观察值，不能证明 follow-up 带来固定收益。")


def main():
    wait_token_demo()


if __name__ == "__main__":
    main()
    print("OK")
