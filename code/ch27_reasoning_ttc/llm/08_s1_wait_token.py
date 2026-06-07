# ---
# chapter: 27
# topic: s1 Wait token extension + budget extrapolation
# section: 27.3.3 s1
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 08_s1_wait_token.py
# expected_runtime: <1s
# expected_output: 打印不同 wait 策略下思维链延长比例
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3.3
# Interview hooks:
#   1. 为什么注入 "Wait" 单一 token 就能让模型"再多想想"？
#   2. 训练 s1.1 时如何构造"模型提前想给答案"的样本？
#   3. budget extrapolation 极限在哪？何时会反噬？
"""Wait token 注入：把"提前终止"的回答硬拉回思考阶段。

s1 / s1.1 核心机制：
  模型 fine-tune 学会在长 CoT 末尾用 "Wait" 自我打断。
  推理时若检测到 < budget 就出现 "Therefore"，
  强制注入 "Wait" 让它续写。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


WAIT_PHRASES = [
    "Wait", "But wait", "Hmm, let me reconsider",
    "Actually, I need to verify", "Hold on",
]

# 模型"想要"给答案的信号
FINAL_SIGNAL = re.compile(
    r"\b(therefore|so the answer is|thus,? we have|finally)\b", re.I
)


@dataclass
class WaitStats:
    n_injections: int
    original_thought_tokens: int
    final_thought_tokens: int
    extension_ratio: float


def extend_with_wait(
    stream_tokens: list[str],
    budget: int,
    patience: int = 256,
) -> tuple[list[str], WaitStats]:
    """边生成边检测：若距离 budget 还剩 > patience 且出现 final signal，
    就注入一个 Wait phrase。
    """
    out: list[str] = []
    n_waits = 0
    for tok in stream_tokens:
        out.append(tok)
        joined = " ".join(out)
        if (len(out) < budget - patience
                and FINAL_SIGNAL.search(joined)):
            out.append("\n" + WAIT_PHRASES[n_waits % len(WAIT_PHRASES)] + ",")
            n_waits += 1

    # 简单截断到 budget
    if len(out) > budget:
        out = out[:budget]

    stats = WaitStats(
        n_injections=n_waits,
        original_thought_tokens=min(len(stream_tokens), budget),
        final_thought_tokens=len(out),
        extension_ratio=len(out) / max(1, min(len(stream_tokens), budget)),
    )
    return out, stats


def main() -> None:
    # 模拟模型输出流：包含"提前 final"信号
    raw = (
        "x is rational, so x = p/q, p^2 = 2q^2, "
        "Therefore the answer is that sqrt(2) is rational."  # 错的！
    ).split()
    print(f"原始 thought tokens: {len(raw)}\n")

    # 策略 1: 不注入
    out1, s1 = extend_with_wait(raw, budget=2000, patience=999_999)
    print(f"[no-wait]   injections={s1.n_injections}  "
          f"final_tokens={s1.final_thought_tokens}  "
          f"extension={s1.extension_ratio:.2f}x")

    # 策略 2: patience=200 → 触发注入
    out2, s2 = extend_with_wait(raw, budget=2000, patience=20)
    print(f"[p=20]      injections={s2.n_injections}  "
          f"final_tokens={s2.final_thought_tokens}  "
          f"extension={s2.extension_ratio:.2f}x")

    # 策略 3: 多次注入直到 budget 满
    out3, s3 = extend_with_wait(raw, budget=2000, patience=10)
    print(f"[p=10]      injections={s3.n_injections}  "
          f"final_tokens={s3.final_thought_tokens}  "
          f"extension={s3.extension_ratio:.2f}x")

    # 演示：超过 budget 后强制截断
    print(f"\n--- extended thought preview (p=20) ---")
    print(" ".join(out2[:60]), "...")

    print("\n关键:")
    print("  • patience 越小 → 注入越频繁 → 思维链越长")
    print("  • budget 是硬上限 → 超过则截断 (会有答案缺失风险)")
    print("  • s1.1: 用更强的 base 模型 + 更大 budget 上限")


if __name__ == "__main__":
    main()
