# ---
# chapter: 27
# topic: RLVR — Reinforcement Learning with Verifiable Rewards
# section: 27.6.2 RL 阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 10_rlvr_rewards.py
# expected_runtime: <1s
# expected_output: 演示 math / code / format 三类可验证奖励
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.6.2
# Interview hooks:
#   1. 什么是 RLVR？和 RLHF 的本质区别？
#   2. 数学/代码/事实型任务如何构造"零成本 verifier"？
#   3. RLVR 局限：哪些任务无法 verifiable？(开放问答、创意写作)
"""RLVR：用规则/单元测试/数学等价验证替代人类偏好 reward model。

DeepSeek-R1 / Tulu 3 / OpenAI o 系列核心训练信号。
无需 RM → 避免 reward hacking → 训练更稳。
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable


@dataclass
class VerifierResult:
    score: float        # 0.0 ~ 1.0
    details: str
    passed: bool


# ---------- 1. 数学 verifier ----------
def math_verifier(answer: str, ground_truth: str) -> VerifierResult:
    """从 answer 中提取 \boxed{...} 内容，做归一化比较。"""
    m = re.search(r"\\boxed\{([^}]+)\}", answer)
    if not m:
        return VerifierResult(0.0, "no \\boxed{}", False)
    pred = m.group(1).strip().rstrip(".")
    gt = ground_truth.strip().rstrip(".")
    # 数字归一化
    try:
        ok = abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        ok = pred.lower() == gt.lower()
    return VerifierResult(1.0 if ok else 0.0,
                          f"pred={pred!r}  gt={gt!r}", ok)


# ---------- 2. 代码 verifier (sandbox 跑单测) ----------
def code_verifier(code: str, tests: list[str], timeout: float = 2.0) -> VerifierResult:
    """执行用户代码 + 单测，返回通过率。生产环境用 Docker/Firejail 隔离。"""
    full = code + "\n\n" + "\n".join(tests)
    n_pass = 0
    n_total = len(tests)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        out = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=timeout
        )
        # 简化：把每个 assert 拆开看错误
        for t in tests:
            single = code + "\n" + t
            r = subprocess.run(["python", "-c", single],
                               capture_output=True, timeout=timeout)
            if r.returncode == 0:
                n_pass += 1
    except subprocess.TimeoutExpired:
        return VerifierResult(0.0, "timeout", False)
    except Exception as e:
        return VerifierResult(0.0, f"err: {e}", False)
    score = n_pass / max(1, n_total)
    return VerifierResult(score, f"{n_pass}/{n_total} tests passed",
                          score == 1.0)


# ---------- 3. 格式 verifier ----------
def format_verifier(answer: str) -> VerifierResult:
    """检查是否包含 <think> 块 + 答案格式。"""
    has_think = bool(re.search(r"<think>.*?</think>", answer, re.S))
    has_boxed = bool(re.search(r"\\boxed\{", answer))
    score = 0.5 * has_think + 0.5 * has_boxed
    return VerifierResult(score, f"think={has_think} boxed={has_boxed}",
                          score == 1.0)


# ---------- 组合 ----------
def rlvr_reward(answer: str, task: dict) -> float:
    """总 reward = w_format * format + w_correct * verifier。"""
    w_fmt, w_corr = 0.1, 0.9
    f = format_verifier(answer)
    if task["type"] == "math":
        c = math_verifier(answer, task["gt"])
    elif task["type"] == "code":
        c = code_verifier(answer, task["tests"])
    else:
        c = VerifierResult(0.0, "unknown task", False)
    return w_fmt * f.score + w_corr * c.score


def main() -> None:
    # Math 任务
    math_task = {
        "type": "math",
        "question": "2+2=?",
        "gt": "4",
    }
    answers = [
        "<think>2+2=4</think>The answer is \\boxed{4}.",
        "<think>2+2=5</think>The answer is \\boxed{5}.",  # 错
        "The answer is 4.",  # 无 think
    ]
    print("=== Math RLVR ===")
    for a in answers:
        v = math_verifier(a, math_task["gt"])
        print(f"  score={v.score}  {v.details}")

    # Code 任务
    code_task = {
        "type": "code",
        "question": "implement add(a,b)",
        "tests": [
            "assert add(1,2) == 3",
            "assert add(-1,1) == 0",
            "assert add(0,0) == 0",
        ],
    }
    print("\n=== Code RLVR ===")
    samples = [
        "def add(a,b): return a+b",                           # 全对
        "def add(a,b): return a-b",                           # 错
        "def add(a,b):\n    while True: pass",                 # 死循环
    ]
    for s in samples:
        v = code_verifier(s, code_task["tests"])
        print(f"  score={v.score:.2f}  {v.details}")

    # 综合
    print("\n=== Combined reward (math task) ===")
    for a in answers:
        r = rlvr_reward(a, math_task)
        print(f"  R={r:.2f}  answer={a[:50]!r}")

    print("OK")


if __name__ == "__main__":
    main()
