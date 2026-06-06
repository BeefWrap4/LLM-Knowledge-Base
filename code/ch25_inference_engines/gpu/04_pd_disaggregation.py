# ---
# chapter: 25
# topic: Prefill-Decode Disaggregation
# section: 25.4
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 04_pd_disaggregation.py
# expected_runtime: <1s
# expected_output: 模拟 PD-Disagg 调度器：prefill 池 + decode 池，KV 通过 RDMA-like 通道传输
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.4
# Interview hooks:
#   1. 为什么要把 prefill 和 decode 拆到不同的实例上？
#   2. PD-Disagg 的关键瓶颈是什么？(答: KV transfer, 需 NVLink/IB/RDMA)
#   3. 什么场景 PD-Disagg 收益最大？(答: 长 prompt + 短 output、解码阶段高并发)

"""Mock PD-Disaggregation scheduler.

Two pools: prefill instances (compute-bound) and decode instances
(memory-bound). When prefill finishes for a request, its KV cache
is *transferred* to a decode instance which then runs the
auto-regressive step.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Request:
    rid: int
    prompt_len: int
    out_len: int
    generated: int = 0
    kv_bytes: int = 0
    state: str = "queued"  # queued | prefilling | transferring | decoding | done
    started_at: float = 0.0
    prefill_done_at: float = 0.0
    first_token_at: float = 0.0
    finished_at: float = 0.0

    def ttft(self) -> float:
        if self.first_token_at == 0:
            return -1.0
        return self.first_token_at - self.started_at


@dataclass
class Instance:
    name: str
    capacity: int
    role: str  # "prefill" or "decode"
    queue: list[Request] = field(default_factory=list)
    busy_slots: int = 0


class PDScheduler:
    KV_PER_TOKEN_BYTES = 2 * 80 * 128 * 2  # 2 (K+V) × layers × head_dim × fp16

    def __init__(self, n_prefill: int = 2, n_decode: int = 4,
                 prefill_cap: int = 4, decode_cap: int = 8,
                 prefill_step_ms: float = 50.0, decode_step_ms: float = 20.0,
                 transfer_gbps: float = 200.0) -> None:
        self.prefill_pool = [Instance(f"prefill-{i}", prefill_cap, "prefill")
                             for i in range(n_prefill)]
        self.decode_pool = [Instance(f"decode-{i}", decode_cap, "decode")
                            for i in range(n_decode)]
        self.prefill_step_ms = prefill_step_ms
        self.decode_step_ms = decode_step_ms
        self.transfer_gbps = transfer_gbps

    # ---- scheduling primitives ----
    def _pick(self, pool: list[Instance]) -> Optional[Instance]:
        for inst in pool:
            if inst.busy_slots < inst.capacity:
                return inst
        return None

    def admit(self, req: Request, t: float) -> None:
        req.started_at = t
        inst = self._pick(self.prefill_pool)
        if inst is None:
            return  # back-pressure
        inst.queue.append(req)
        inst.busy_slots += 1
        req.state = "prefilling"

    def step(self, t: float, dt_ms: float) -> None:
        # 1) prefill progress
        for inst in self.prefill_pool:
            for req in inst.queue:
                if req.state != "prefilling":
                    continue
                # In prefill, work is roughly linear in prompt_len (with batching gains)
                prefill_ms = req.prompt_len / 1000.0 * self.prefill_step_ms
                if dt_ms >= prefill_ms:
                    req.kv_bytes = req.prompt_len * self.KV_PER_TOKEN_BYTES
                    req.state = "transferring"
                    req.prefill_done_at = t
                    # In real systems: hand off to transfer queue (RDMA/NVLink)
                    transfer_ms = (req.kv_bytes * 8) / (self.transfer_gbps * 1e9) * 1000
                    req.first_token_at = t + transfer_ms
            # clear completed prefill slots
            inst.queue = [r for r in inst.queue if r.state == "prefilling"]
            inst.busy_slots = len(inst.queue)

        # 2) transfer queue → decode
        for req in [r for inst in self.prefill_pool for r in inst.queue]:
            if req.state != "transferring":
                continue
            dinst = self._pick(self.decode_pool)
            if dinst is None:
                continue
            dinst.queue.append(req)
            dinst.busy_slots += 1
            req.state = "decoding"

        # 3) decode step
        for inst in self.decode_pool:
            for req in inst.queue:
                if req.state != "decoding":
                    continue
                req.generated += 1
                if req.generated >= req.out_len:
                    req.state = "done"
                    req.finished_at = t
                    inst.busy_slots -= 1
            inst.queue = [r for r in inst.queue if r.state == "decoding"]


def main() -> None:
    sched = PDScheduler()
    work = [
        Request(rid=i, prompt_len=p, out_len=o)
        for i, (p, o) in enumerate([(2048, 64), (1024, 128), (4096, 32), (512, 200), (8192, 16)])
    ]
    sim_t = 0.0
    for r in work:
        sched.admit(r, sim_t)
    dt = 50.0
    for tick in range(120):
        sched.step(sim_t, dt)
        sim_t += dt / 1000.0
        if all(r.state == "done" for r in work):
            print(f"all done at t={sim_t:.2f}s")
            break

    print("\nper-request TTFT (s) and makespan (s):")
    for r in work:
        print(f"  req {r.rid}: TTFT={r.ttft():.3f}  "
              f"total={r.finished_at - r.started_at:.3f}  "
              f"kv_MB={r.kv_bytes/1e6:.1f}")
    # Key insight: long-prompt reqs pay big prefill cost, but their decode
    # step starts on a separate, memory-optimized instance.
    print("OK")


if __name__ == "__main__":
    main()
