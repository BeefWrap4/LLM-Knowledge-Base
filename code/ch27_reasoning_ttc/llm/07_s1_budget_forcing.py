# ---
# chapter: 27
# topic: s1 simple test-time scaling + budget forcing
# section: 27.3.3 s1 (Stanford 2025)
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0, DEEPSEEK_API_KEY
# run: python 07_s1_budget_forcing.py
# expected_runtime: <120s (real DeepSeek V4 Pro with bounded follow-up loops)
# expected_output: prints API approximation rounds/reasoning chars + final answer
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3.3
# Interview hooks:
#   1. Budget forcing 与 RL 训练的 cost model 区别？
#   2. "Wait" token 触发的训练时分布偏移如何缓解？
#   3. 为什么 Chat Completions 多轮提示不能等同于 s1 的 token 级 budget forcing？
"""S1: Simple Test-Time Scaling 与托管 API 近似演示.

S1 核心: 控制推理时"思考 token 数", 强制模型用尽/截断思考:
  - 强制等长: 追加 "Wait" 触发继续思考
  - 强制截断: 追加最终答案 marker (如 "Final Answer:")

DeepSeek 托管 Chat Completions API 不开放 token 级解码控制。本例只能用
后续 user turn 请求复核，并统计返回的 reasoning_content 字符数；它不是
s1 论文方法的严格复现，也不能把字符数当作 token budget。

参考: "s1: Simple test-time scaling" (arXiv 2501.19393) 2025
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


def budget_forcing_api_approximation(
    client,
    model: str,
    prompt: str,
    target_reasoning_chars: int = 2000,
) -> dict:
    """用有限多轮复核近似展示预算思想，不宣称控制真实 reasoning tokens.

    流程:
      1. 调用 V4 thinking mode，读取 reasoning_content 与最终回答
      2. 若累计 reasoning 字符数仍低于演示阈值，追加 user 复核请求
      3. 达到字符阈值或最大轮数后停止

    真实 s1 budget forcing 需要能控制模型解码流/停止位置的本地推理接口。
    """

    messages = [{"role": "user", "content": prompt}]
    reasoning_chunks: list[str] = []
    final_content = ""
    iterations = 0
    max_iterations = 3

    while iterations < max_iterations:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", "") or ""
        final_content = msg.content or ""
        reasoning_chunks.append(reasoning)
        iterations += 1

        if sum(map(len, reasoning_chunks)) >= target_reasoning_chars:
            break

        # 非工具多轮不需要回传隐藏思考；只保留可见答案，再用新 user turn 请求复核。
        messages.append({"role": "assistant", "content": final_content or "(尚未给出最终答案)"})
        messages.append(
            {
                "role": "user",
                "content": "Wait. Re-check the assumptions and solution, then give a corrected final answer.",
            }
        )

    return {
        "reasoning_chars": sum(map(len, reasoning_chunks)),
        "iterations": iterations,
        "final": final_content,
    }


def main():
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("=== s1 budget forcing（DeepSeek V4 API 离线演示）===")
        print("model=deepseek-v4-pro, thinking=enabled, reasoning_effort=high")
        print("托管 API 只能做多轮复核近似；严格复现需本地 token 级解码控制。")
        return

    api_key = get_deepseek_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=== s1 Budget Forcing（DeepSeek V4 API 近似）===\n")

    result = budget_forcing_api_approximation(
        client,
        "deepseek-v4-pro",
        "9.11 和 9.9 哪个更大? 详细推理",
        target_reasoning_chars=2000,
    )

    print(f"  returned reasoning_content: {result['reasoning_chars']} chars")
    print(f"  iterations: {result['iterations']}")
    print(f"\n  final: {result['final'][:300]}")
    print("\n  边界: 字符数不是 token 数，多轮复核也不是严格的 s1 budget forcing。")


if __name__ == "__main__":
    main()
    print("OK")
