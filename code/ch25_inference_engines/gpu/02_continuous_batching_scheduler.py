# ---
# chapter: 25
# topic: Continuous Batching Scheduler
# section: 25.2.1 / 25.4
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 02_continuous_batching_scheduler.py
# expected_runtime: <1s
# expected_output: 演示 static batching vs continuous batching 的吞吐差异
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.1
# Interview hooks:
#   1. 什么是 continuous batching？和 static batching 比有什么优势？
#   2. Iteration-level scheduling 解决了什么问题？(答: padding 浪费、head-of-line blocking)
#   3. vLLM 如何在每个 decode step 决定插入/驱逐哪些请求？

"""Continuous batching vs static batching (simulation)."""
from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    prompt_len: int
    out_len: int
    generated: int = 0

    @property
    def is_done(self) -> bool:
        return self.generated >= self.out_len

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated


def gen_workload(n: int) -> list[Req]:
    rng = random.Random(0)
    return [
        Req(rid=i, prompt_len=rng.randint(50, 500), out_len=rng.randint(20, 200))
        for i in range(n)
    ]


def run_static(reqs: list[Req], max_batch: int) -> tuple[int, int, float]:
    """Wait for batch to fill, run until ALL in batch done. Classic HF/pipeline style."""
    pending = list(reqs)
    steps_done = 0
    tokens = 0
    util_tokens = 0
    total_len = sum(r.total_len for _ in [0] for r in reqs)  # sum of final lens
    # Batches are formed greedily
    while pending:
        batch = pending[:max_batch]
        pending = pending[max_batch:]
        # Within a batch, find the longest output needed and step everyone together
        for step in range(max(r.out_len for r in batch)):
            still_running = [r for r in batch if not r.is_done]
            if not still_running:
                break
            # every active request produces 1 token
            for r in still_running:
                r.generated += 1
            steps_done += 1
            tokens += len(still_running)
            # padding = all - active  (in real systems, this is wasted compute)
            util_tokens += len(still_running)
    utilization = util_tokens / max(1, tokens + util_tokens * 0)  # simple proxy
    return tokens, steps_done, utilization


def run_continuous(reqs: list[Req], max_batch: int) -> tuple[int, int]:
    """Iteration-level: at every step, swap finished seqs with waiting ones."""
    waiting = list(reqs)
    running: list[Req] = []
    steps = 0
    tokens = 0
    while waiting or running:
        # fill up to max_batch
        while waiting and len(running) < max_batch:
            running.append(waiting.pop(0))
        if not running:
            break
        # one decode step for everyone
        for r in running:
            r.generated += 1
        steps += 1
        tokens += len(running)
        # evict finished, admit new (key advantage)
        running = [r for r in running if not r.is_done]
    return tokens, steps


def main() -> None:
    reqs = gen_workload(40)

    static_tokens, static_steps, _ = run_static(reqs, max_batch=8)
    cont_tokens, cont_steps = run_continuous(reqs, max_batch=8)

    print(f"static batching     : steps={static_steps:4d}  tokens={static_tokens}")
    print(f"continuous batching : steps={cont_steps:4d}  tokens={cont_tokens}")
    speedup = static_steps / max(1, cont_steps)
    print(f"step-count ratio    : static={static_steps}  continuous={cont_steps}")
    # The big wins are (a) no padding waste, (b) head-of-line blocking gone.
    print("OK")


if __name__ == "__main__":
    main()
