# ---
# chapter: 27
# topic: s1 simple test-time scaling + budget forcing
# section: 27.3.3 s1 (Stanford 2025)
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0, DEEPSEEK_API_KEY
# run: python 07_s1_budget_forcing.py
# expected_runtime: <120s (real DeepSeek R1 with forced wait loops)
# expected_output: prints budget-forced reasoning length + final answer
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3.3
# Interview hooks:
#   1. Budget forcing 与 RL 训练的 cost model 区别？
#   2. "Wait" token 触发的训练时分布偏移如何缓解？
#   3. S1 在 AIME 24 上相对 base R1 提升多少？计算量/准确率曲线斜率？
"""S1: Simple Test-Time Scaling (Budget Forcing).

S1 核心: 控制推理时"思考 token 数", 强制模型用尽/截断思考:
  - 强制等长: 追加 "Wait" 触发继续思考
  - 强制截断: 追加最终答案 marker (如 "Final Answer:")

参考: "Simple Test-Time Scaling" (arXiv 2501.19308) 2025
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


def budget_forced_generation(client, model: str, prompt: str, max_budget: int = 2000) -> dict:
    """S1 budget forcing: 强制模型用足 / 截断 reasoning tokens.

    流程:
      1. 第一次调 API, 拿到 partial response (含 reasoning)
      2. 如 reasoning < max_budget, 追加 "Wait" 强制继续
      3. 重复, 直至 reasoning >= max_budget 或达到 N 次迭代
    """
    from openai import OpenAI

    messages = [{"role": "user", "content": prompt}]
    total_reasoning = ""
    final_content = ""
    iterations = 0
    max_iterations = 5

    while iterations < max_iterations:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=4096,
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", "") or ""
        final_content = msg.content or ""
        total_reasoning += reasoning
        iterations += 1

        if len(total_reasoning) >= max_budget:
            break
        if not final_content:  # 模型还没给最终答案, 强制继续
            messages.append({"role": "assistant", "content": total_reasoning})
            messages.append({"role": "user", "content": "Wait, continue thinking."})
        else:
            break

    return {
        "reasoning_len": len(total_reasoning),
        "iterations": iterations,
        "final": final_content,
    }


def main():
    api_key = get_deepseek_key()

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    print("=== S1 Budget Forcing (DeepSeek-R1) ===\n")

    result = budget_forced_generation(
        client, "deepseek-reasoner",
        "9.11 和 9.9 哪个更大? 详细推理",
        max_budget=2000,
    )

    print(f"  reasoning length: {result['reasoning_len']} chars")
    print(f"  iterations: {result['iterations']}")
    print(f"\n  final: {result['final'][:300]}")
    print(f"\n  S1 关键:")
    print(f"    - max_budget: 强制模型推理 token 数")
    print(f"    - Wait token: 触发继续思考")
    print(f"    - Final Answer: 触发截断")


if __name__ == "__main__":
    main()
