# ---
# chapter: 16
# topic: 数学表达式验证器 (RLVR Verifier, 纯 stdlib)
# section: 16.9.4
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: (stdlib only - ast, re, fractions)
# run: python 14_math_verifier.py
# expected_runtime: <1s
# expected_output: 7 个测试用例的 verify 结果
# ---
# See: ../../../16_模型微调与推理优化.md §16.10.4
#
# Interview hooks:
#   1. Verifier 可重复为什么不等于“奖励无噪声”？
#   2. 数学题 verifier 的容错设计：符号等价 / 数值等价 / 表达式解析？
#   3. 复合奖励 (准确性 + 格式) 的工程价值？避免模型"猜答案"投机？
"""RLVR (RL with Verifiable Rewards) 数学表达式验证器.

程序化 verifier 可以重复执行，但其规格、解析器、标准答案与测试覆盖仍可能有误，也可能被
策略利用。本例只覆盖有限算术语法，不代表数学证明验证器。

支持:
  - 基本运算 + - * /
  - 分数 (Fraction, 精确小数)
  - ast 解析 (无 eval 注入风险)
  - 字符串 + 数值双轨比对
"""

import ast
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))


def safe_eval(expr: str) -> float:
    """用 ast 安全求值, 无内置 eval 注入风险."""
    if len(expr) > 128:
        raise ValueError("表达式过长")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if type(node.value) in {int, float}:
            if not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError("数值超出教学验证器范围")
            return node.value
        raise ValueError(f"不支持的字面量: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, int) or abs(right) > 12:
                raise ValueError("指数超出教学验证器范围")
            return left**right
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _eval_node(node.operand)
    raise ValueError(f"不支持的运算: {type(node).__name__}")


def normalize_answer(ans: str) -> str:
    """规范化答案: 去空格/尾部标点."""
    ans = ans.strip()
    ans = re.sub(r"\s+", "", ans)
    ans = ans.rstrip("。.,;；:：")
    return ans


def verify(expected: str, predicted: str, tol: float = 1e-6) -> bool:
    """验证 predicted 与 expected 是否等价 (字符串或数值)."""
    e_norm = normalize_answer(expected)
    p_norm = normalize_answer(predicted)
    if e_norm == p_norm:
        return True
    # 数值比较 (Fraction 精确)
    try:
        e_val = Fraction(safe_eval(expected))
        p_val = Fraction(safe_eval(predicted))
        return abs(float(e_val - p_val)) < tol
    except (OverflowError, TypeError, ValueError, SyntaxError, ZeroDivisionError):
        return False


def composite_reward(model_output: str, ground_truth: str) -> tuple[float, dict]:
    """RLVR 复合奖励: 准确性 + 格式 + 过程标记."""
    # 1) 准确性: 提取 \\boxed{...} 答案
    match = re.search(r"\\boxed\{([^{}]+)\}", model_output)
    has_boxed = match is not None
    if not has_boxed:
        acc = 0.0
    else:
        acc = 1.0 if verify(match.group(1), ground_truth) else 0.0
    # 2) 格式: 思考块
    has_think = "<think>" in model_output and "</think>" in model_output
    # 3) 复合: accuracy + 格式 bonus
    fmt_bonus = 0.1 if (has_boxed and has_think) else 0.0
    total = acc + fmt_bonus
    return total, {
        "accuracy": acc,
        "has_boxed": has_boxed,
        "has_think": has_think,
        "format_bonus": fmt_bonus,
        "total": total,
    }


def main():
    print("=== RLVR 数学表达式验证器 ===\n")
    print("[1/2] verify() 单元测试:\n")
    test_cases = [
        ("1 + 2 * 3", "7", True),
        ("(1+2)*3", "9", True),
        ("100/4", "25", True),
        ("1/3 + 1/3 + 1/3", "1", True),
        ("3.14 * 2", "6.28", True),
        ("1 + 1", "2.0", True),  # 字符串不同但数值等
        ("1 + 1", "3", False),  # 数值不等
    ]
    correct = 0
    for expected, predicted, want in test_cases:
        got = verify(expected, predicted)
        ok = got == want
        if ok:
            correct += 1
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] verify({expected!r}, {predicted!r}) = {got}  (期望 {want})")
    print(f"\n  {correct}/{len(test_cases)} 通过\n")

    print("[2/2] composite_reward() 复合奖励（格式 bonus 不证明过程正确）:\n")
    rlvr_cases = [
        ("<think>因式分解 (x-2)(x-3)...</think>答案是 \\boxed{2, 3}", "2, 3"),
        ("<think>...</think>\\boxed{1, 4}", "2, 3"),  # 错答案
        ("答案是 2 和 3 (无 boxed)", "2, 3"),  # 无格式
        ("\\boxed{6.0}", "6"),  # 数值等价
    ]
    for output, gt in rlvr_cases:
        total, info = composite_reward(output, gt)
        print(
            f"  total={total:.2f} | acc={info['accuracy']} "
            f"| boxed={info['has_boxed']} | think={info['has_think']}"
        )
        print(f"    pred='{output[:60]}...' gt='{gt}'")
    print("\n  注意: verifier 仍需规格审查、对抗测试和人工抽检")
    print("OK")


if __name__ == "__main__":
    main()
