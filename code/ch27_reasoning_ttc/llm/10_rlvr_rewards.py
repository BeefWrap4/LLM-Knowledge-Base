# ---
# chapter: 27
# topic: RLVR — Reinforcement Learning with Verifiable Rewards
# section: 27.6.2 RL 阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 10_rlvr_rewards.py
# expected_runtime: <1s (regex + math)
# expected_output: prints 3 case rewards (correct/wrong/bad-format)
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.6.2
# Interview hooks:
#   1. 什么是 RLVR？和 RLHF 的本质区别？
#   2. 数学/代码/事实型任务如何构造"零成本 verifier"？
#   3. RLVR 局限：哪些任务无法 verifiable？(开放问答、创意写作)
"""RLVR (RL from Verifiable Rewards) Reward 函数.

RLVR 用可验证的 reward (vs learned reward model):
  - math: 答案是否正确
  - code: 单元测试是否通过
  - format: 输出格式是否合规 (think + answer 块)
"""

import re
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))


def math_reward(predicted: str, ground_truth: str) -> float:
    """数学答案 reward: 0 或 1."""
    # 简单: 提取数字比较
    pred_nums = re.findall(r"-?\d+\.?\d*", predicted)
    truth_nums = re.findall(r"-?\d+\.?\d*", ground_truth)
    if pred_nums and truth_nums:
        try:
            return 1.0 if abs(float(pred_nums[-1]) - float(truth_nums[-1])) < 1e-6 else 0.0
        except ValueError:
            return 0.0
    return 0.0


def code_reward(code: str, tests_pass: bool) -> float:
    """代码 reward: 单元测试通过 = 1, 否则 0."""
    return 1.0 if tests_pass else 0.0


def format_reward(output: str, require_think: bool = True) -> float:
    """格式 reward: think + answer 块结构合规."""
    if require_think:
        # 要求: <think>...</think>\nAnswer: ...
        has_think = bool(re.search(r"<think>.*?</think>", output, re.DOTALL))
        has_answer = "Answer:" in output or "Final Answer:" in output
        return 0.5 if (has_think and has_answer) else 0.0
    return 1.0


def composite_reward(
    predicted: str,
    ground_truth: str = "",
    code: str = "",
    tests_pass: bool = False,
    weights: dict = None,
) -> float:
    """组合 reward = w1 * math + w2 * code + w3 * format."""
    if weights is None:
        weights = {"math": 0.6, "code": 0.3, "format": 0.1}
    r_math = math_reward(predicted, ground_truth) if ground_truth else 0
    r_code = code_reward(code, tests_pass) if code else 0
    r_format = format_reward(predicted, require_think=True)

    total = weights["math"] * r_math + weights["code"] * r_code + weights["format"] * r_format
    return total


def main():
    print("=== RLVR Reward Functions ===\n")

    # 数学题
    output1 = "<think> 9.11 > 9.9 because 9.11 = 9 + 11/100 > 9 + 9/10 = 9.9 </think>\nAnswer: 9.11"
    r1 = composite_reward(output1, ground_truth="9.11")
    print(f"  Case 1 (correct math + good format): reward = {r1:.3f}")

    # 错误答案
    output2 = "<think> thinking... </think>\nAnswer: 9.9"
    r2 = composite_reward(output2, ground_truth="9.11")
    print(f"  Case 2 (wrong math + good format):  reward = {r2:.3f}")

    # 缺 think
    output3 = "Answer: 9.11"
    r3 = composite_reward(output3, ground_truth="9.11")
    print(f"  Case 3 (correct + bad format):       reward = {r3:.3f}")

    print("\n  RLVR 优势:")
    print("    - 不需要 learned reward model (节省训练)")
    print("    - 客观可验证, 避免 reward hacking")
    print("    - 适合 math / code / factual QA")


if __name__ == "__main__":
    main()
