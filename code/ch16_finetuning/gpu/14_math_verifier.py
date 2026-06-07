# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.11.4 RLVR Verifier 编写示例
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only - re)
# run: python 14_math_verifier.py
# expected_runtime: <1s
# expected_output: 4 个测试用例的 reward 评估结果 (正确/错误/格式错误)
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.4
# Interview hooks:
#   1. RLVR 与传统 RLHF 的核心区别？奖励噪声对比？
#   2. 数学题 verifier 的容错设计：符号等价 / 数值等价 / 表达式解析？
#   3. 复合奖励（准确性 + 格式）的工程价值？避免模型"猜答案"投机？

"""
RLVR (RL with Verifiable Rewards) - 数学题 verifier 示例

奖励来自可自动验证的程序 (非人类偏好), 几乎零噪声
"""

import re
from typing import Tuple


def math_reward(model_output: str, ground_truth: str) -> float:
    """
    检查模型最终答案是否在 \boxed{} 中且等于标准答案
    返回 0.0 / 1.0
    """
    # 1) 提取 \boxed{...} 中的答案
    match = re.search(r"\\boxed\{([^{}]+)\}", model_output)
    if not match:
        return 0.0
    pred = match.group(1).strip()
    # 2) 简单字符串等价 + 数值等价
    if pred == ground_truth.strip():
        return 1.0
    try:
        if abs(float(pred) - float(ground_truth)) < 1e-6:
            return 1.0
    except ValueError:
        pass
    return 0.0


def check_format(model_output: str) -> bool:
    """检查是否包含 <think>...</think> 思考块 + 最终 \\boxed{} 答案"""
    has_think = "<think>" in model_output and "</think>" in model_output
    has_boxed = "\\boxed{" in model_output
    return has_think and has_boxed


def composite_reward(model_output: str, gt: str) -> Tuple[float, dict]:
    """
    复合奖励 = 准确性奖励 + 格式奖励
    """
    acc = math_reward(model_output, gt)
    fmt = check_format(model_output)
    fmt_bonus = 0.1 if fmt else 0.0
    total = acc + fmt_bonus
    info = {
        "accuracy": acc,
        "format_ok": fmt,
        "format_bonus": fmt_bonus,
        "total": total,
    }
    return total, info


if __name__ == "__main__":
    test_cases = [
        # 正确答案 + 正确格式
        (
            "<think> 先因式分解 x^2-5x+6 = (x-2)(x-3)... </think>\n"
            "所以 x=2 或 x=3, 最终答案 \\boxed{2, 3}",
            "2, 3",
        ),
        # 错误答案
        (
            "<think> ... </think> \\boxed{1, 4}",
            "2, 3",
        ),
        # 格式错误 (无 boxed)
        (
            "答案是 2 和 3",
            "2, 3",
        ),
        # 数值等价 (字符串不同但数值相同)
        (
            "\\boxed{6.0}",
            "6",
        ),
    ]

    for i, (output, gt) in enumerate(test_cases, 1):
        total, info = composite_reward(output, gt)
        print(f"Case {i}: total={total:.2f} | acc={info['accuracy']} | "
              f"format_ok={info['format_ok']}")
        print(f"  pred='{output[:50]}...' gt='{gt}'")
