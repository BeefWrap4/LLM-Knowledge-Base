# ---
# chapter: 27
# topic: s1 simple test-time scaling + budget forcing
# section: 27.3.3 s1 (Stanford 2025)
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 07_s1_budget_forcing.py
# expected_runtime: <1s
# expected_output: 打印 budget 强制生成 + 答案提取
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.3.3
# Interview hooks:
#   1. s1 的核心思想是什么？为什么 1K 样本就能让 32B 模型学会长 CoT？
#   2. budget forcing 的实现方式（截断 / 注入 wait token）？
#   3. s1 与 DeepSeek-R1 的差异？(数据规模 + RL)
"""s1: 用 1K 高质量长 CoT 样本 SFT + 推理时 budget forcing。

两个核心技巧:
  1. 用特殊分隔符 `<|im_start|>think ... <|im_end|>` 强制结束思考
  2. 当模型提前想给答案时，追加 "Wait" token 让它继续
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# s1 论文推荐的特殊 token
THINK_END = "<|im_start|>answer"
WAIT_TOKEN = "Wait"


@dataclass
class S1Config:
    max_think_tokens: int = 4000   # 论文默认 budget
    max_answer_tokens: int = 2000
    inject_wait: bool = True
    wait_patience: int = 200       # 距离答案 < 200 tok 时注入 Wait


@dataclass
class S1Output:
    think: str
    answer: str
    n_wait_injections: int


def budget_force(raw_generation: str, cfg: S1Config) -> S1Output:
    """把模型原始输出切分为 think / answer，并按 budget 强制结束。

    Mock：把传入的 raw_generation 视为完整长 CoT。
    """
    # 1) 切分 think 与 answer
    if THINK_END in raw_generation:
        think, _, answer = raw_generation.partition(THINK_END)
    else:
        # 模型忘了切分 → 整段当 think
        think, answer = raw_generation, ""

    # 2) 截断过长的 think
    think = truncate_to_tokens(think, cfg.max_think_tokens)

    # 3) 模拟 Wait 注入：检测 "Therefore the answer is" 提前出现
    n_waits = 0
    if cfg.inject_wait:
        early = re.search(r"Therefore,?\s*the\s+answer\s+is", think, re.I)
        if early and cfg.max_think_tokens - len(think.split()) < cfg.wait_patience:
            think += f"\n{WAIT_TOKEN}, let me double-check..."
            n_waits += 1

    # 4) answer 后处理
    answer = answer.strip() or "(forced) 42"

    return S1Output(think=think.strip(), answer=answer, n_wait_injections=n_waits)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """粗略按空格切 token（真实环境用 tokenizer）。"""
    toks = text.split()
    if len(toks) <= max_tokens:
        return text
    return " ".join(toks[:max_tokens])


def main() -> None:
    cfg = S1Config()
    print(f"s1 config: budget={cfg.max_think_tokens}, "
          f"wait_patience={cfg.wait_patience}\n")

    # 模拟模型生成（实际是 streaming 边生成边控制）
    raw = (
        "Let me think about this math problem carefully. We have x^2 = 2. "
        "If x is rational, x = p/q, then p^2 = 2 q^2. "
        "This means p^2 is even, so p is even, let p=2k, then 4k^2 = 2 q^2, "
        "q^2 = 2k^2, so q is also even. Contradiction with p/q in lowest terms. "
        "Therefore the answer is: sqrt(2) is irrational. "
        f"{THINK_END} √2 is irrational, proved by infinite descent."
    )
    out = budget_force(raw, cfg)

    print(f"think tokens (approx): {len(out.think.split())}")
    print(f"answer: {out.answer!r}")
    print(f"wait injections: {out.n_wait_injections}")
    print(f"\n--- THINK PREVIEW (first 200 chars) ---\n{out.think[:200]}...")

    # 关键统计：s1.1 vs s1 准确率 vs budget
    print("\ns1 论文结果 (AIME24):")
    print("  budget=4000  → ~56%  (无 Wait 注入)")
    print("  budget=4000 + Wait → ~57%")
    print("  budget=32000 + Wait → ~59%  (s1.1)")

    print("OK")


if __name__ == "__main__":
    main()
