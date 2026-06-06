# ---
# chapter: 25
# topic: TTFT / TPOT SLO Monitor
# section: 25.6
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 11_slo_ttft_tpot_monitor.py
# expected_runtime: <1s
# expected_output: 评估一批推理请求的 TTFT/TPOT 是否满足 SLO
# ---
# See: ../tutorial/25_推理引擎_与高性能服务.md (note typo intentional: corrected below)
# See also: ../tutorial/25_推理引擎与高性能服务.md §25.6
# Interview hooks:
#   1. 什么是 TTFT 和 TPOT？分别由什么决定？
#   2. 如何定义 LLM 服务 SLO？p50/p99 各自代表什么？
#   3. SLO 不达标时如何定位？(答: prefill/ decode 拆分分析、batching 延迟、KV 压力)

"""SLO monitor: TTFT / TPOT for a batch of mock requests."""
from __future__ import annotations
import random
import time
from dataclasses import dataclass


@dataclass
class SLO:
    ttft_p99_ms: float = 300.0     # 99% of requests get first token < 300ms
    tpot_p99_ms: float = 50.0      # 99% of decode steps < 50ms
    e2e_p99_ms: float = 5000.0


@dataclass
class SampledReq:
    rid: int
    prompt_len: int
    out_len: int
    ttft_ms: float
    per_token_ms: list[float]


def simulate(n: int = 200) -> list[SampledReq]:
    rng = random.Random(0)
    out: list[SampledReq] = []
    for i in range(n):
        p = rng.randint(256, 4096)
        o = rng.randint(64, 512)
        # toy model: ttft grows with prompt_len
        ttft = 30 + p * 0.04 + rng.gauss(0, 15)
        # tpot mostly constant, occasionally spikes (memory pressure)
        tpot = [12 + rng.gauss(0, 3) + (20 if rng.random() < 0.05 else 0)
                for _ in range(o)]
        out.append(SampledReq(i, p, o, ttft, tpot))
    return out


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(q * len(s))))
    return s[idx]


def evaluate(reqs: list[SampledReq], slo: SLO) -> dict:
    ttfts = [r.ttft_ms for r in reqs]
    tpots = [t for r in reqs for t in r.per_token_ms]
    e2e = [r.ttft_ms + sum(r.per_token_ms) for r in reqs]
    p = lambda xs, q: percentile(xs, q)
    return {
        "ttft_p50": round(p(ttfts, 0.50), 1),
        "ttft_p99": round(p(ttfts, 0.99), 1),
        "tpot_p50": round(p(tpots, 0.50), 2),
        "tpot_p99": round(p(tpots, 0.99), 2),
        "e2e_p99_ms": round(p(e2e, 0.99), 1),
        "ttft_pass": p(ttfts, 0.99) <= slo.ttft_p99_ms,
        "tpot_pass": p(tpots, 0.99) <= slo.tpot_p99_ms,
        "e2e_pass":  p(e2e, 0.99) <= slo.e2e_p99_ms,
    }


def main() -> None:
    reqs = simulate(500)
    slo = SLO()
    rep = evaluate(reqs, slo)
    print("SLO targets : TTFT<%.0fms  TPOT<%.0fms  E2E<%.0fms"
          % (slo.ttft_p99_ms, slo.tpot_p99_ms, slo.e2e_p99_ms))
    print(f"observed p50/p99:")
    print(f"  TTFT : {rep['ttft_p50']} / {rep['ttft_p99']} ms   "
          f"{'PASS' if rep['ttft_pass'] else 'FAIL'}")
    print(f"  TPOT : {rep['tpot_p50']} / {rep['tpot_p99']} ms   "
          f"{'PASS' if rep['tpot_pass'] else 'FAIL'}")
    print(f"  E2E  : {rep['e2e_p99_ms']} ms                   "
          f"{'PASS' if rep['e2e_pass'] else 'FAIL'}")
    print("OK")


if __name__ == "__main__":
    main()
