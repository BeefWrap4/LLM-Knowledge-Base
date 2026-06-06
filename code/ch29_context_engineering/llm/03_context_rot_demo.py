# ---
# chapter: 29
# topic: Context Rot 现象演示 — 信息在长 context 中后段被"忽略"
# section: 29.3.2
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 03_context_rot_demo.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.3.2
# Cross-refs:
#   - Ch12 Transformer (注意力衰减机制)
#   - Ch14 RAG (为何需要检索而非堆 context)
#
# Interview hooks:
#   - "200K context 是否真的能用?" →  有效长度远小于标称, 中后段质量显著下降
#   - "Context Rot 是什么?"        →  即使模型支持 200K, 实际对中后段关注度仍下降
#   - "如何缓解?"                  →  检索/压缩/Sub-Agent 隔离 context

from __future__ import annotations
import random
from typing import Callable


# 模拟"事实召回"任务: 把 N 个事实塞进 context, 末尾提问其中一个
class RotSimulator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def make_facts(self, n: int) -> list[str]:
        return [f"Fact#{i:04d}: variable_{i} = {self.rng.randint(1, 1000)}" for i in range(n)]

    def attention_weight(self, position: int, total: int) -> float:
        """模拟模型对 context 中第 position 个 token 的"关注度"。
        - 头部 (前 10%): 1.0
        - 中段 (10%-60%): 线性下降到 ~0.7
        - 尾部 (60%-100%): 进一步下降到 ~0.4
        """
        pct = position / total
        if pct < 0.10:
            return 1.0
        if pct < 0.60:
            return 1.0 - (pct - 0.10) / 0.50 * 0.30
        return 0.70 - (pct - 0.60) / 0.40 * 0.30

    def recall_probability(self, position: int, total: int, baseline: float = 0.95) -> float:
        """位置越靠后, 被正确召回的概率越低。"""
        w = self.attention_weight(position, total)
        return baseline * w + 0.05

    def run_experiment(self, total_facts: int, trials: int = 200) -> dict:
        facts = self.make_facts(total_facts)
        # 抽样 4 个位置: 头部 5%, 25%, 75%, 95%
        positions = [int(total_facts * p) for p in [0.05, 0.25, 0.75, 0.95]]
        results = {}
        for p in positions:
            hits = 0
            for _ in range(trials):
                p_eff = self.recall_probability(p, total_facts)
                if self.rng.random() < p_eff:
                    hits += 1
            results[p] = hits / trials
        return results


def run_demo() -> None:
    sim = RotSimulator()
    print("=== Context Rot 实验: 同一召回任务, 事实位于 context 不同位置 ===\n")
    print(f"{'context 大小':<14s} | {'5%(头)':>8s} {'25%':>8s} {'75%':>8s} {'95%(尾)':>8s} | 头尾差")
    print("-" * 60)
    for n in [500, 2_000, 8_000, 32_000, 128_000]:
        r = sim.run_experiment(n, trials=300)
        head, mid, late, tail = r[int(n*0.05)], r[int(n*0.25)], r[int(n*0.75)], r[int(n*0.95)]
        print(f"{n:>10,d}   | {head:>8.2f} {mid:>8.2f} {late:>8.2f} {tail:>8.2f} | {head-tail:+.2f}")
    print("\n观察: 头部召回率显著高于尾部, 体现 Context Rot。")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
