# ---
# chapter: 25
# topic: MoE Expert Parallel Serving
# section: 25.5
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 07_moe_expert_parallel.py
# expected_runtime: <1s
# expected_output: 模拟 Mixtral/DeepSeek-style expert parallel：路由→all-to-all→本地 expert 计算
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.5
# Interview hooks:
#   1. MoE 推理为什么显存省但通信贵？(答: 全部 expert 权重都在显存，token 需 all-to-all)
#   2. Expert Parallel (EP) 和 Tensor Parallel (TP) 的取舍？
#   3. 路由不均衡会怎样？(答: 单 GPU OOM, 可用 expert capacity / drop-and-pad)

"""Mock expert-parallel serving for an 8-expert MoE (Mixtral-style)."""
from __future__ import annotations
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Expert:
    eid: int
    host_rank: int  # which GPU/instance holds this expert
    weight_mb: int = 4000  # 4 GB per expert (fp16)


class MoEEngine:
    def __init__(self, num_experts: int = 8, top_k: int = 2, num_ranks: int = 2) -> None:
        self.top_k = top_k
        self.experts = [Expert(eid=i, host_rank=i % num_ranks) for i in range(num_experts)]
        self.num_ranks = num_ranks
        # in-memory traffic counters
        self.tokens_dispatched = 0
        self.tokens_received = 0

    def route(self, token_id: int) -> list[int]:
        """Mock router: pick top-k experts with skewed distribution."""
        # Hot experts (0, 1) get more traffic
        weights = [0.30, 0.25, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05]
        return sorted(random.choices(range(len(self.experts)), weights=weights, k=self.top_k))

    def step(self, batch: list[int]) -> dict:
        """Simulate one decode step on a batch of tokens.

        For each token:
          1) router picks top_k experts
          2) tokens are *dispatched* to expert hosts (all-to-all traffic)
          3) each rank runs its local experts on the tokens it received
        """
        per_rank: dict[int, list[tuple[int, int]]] = defaultdict(list)
        comm_bytes = 0
        for tok in batch:
            for eid in self.route(tok):
                expert = self.experts[eid]
                per_rank[expert.host_rank].append((tok, eid))
                comm_bytes += 8 * 1024  # ~8 KB per token dispatch (mock)
        # "compute" — for the simulation, we just count
        self.tokens_dispatched += len(batch) * self.top_k
        self.tokens_received += sum(len(v) for v in per_rank.values())
        return {
            "tokens": len(batch),
            "per_rank_load": {r: len(v) for r, v in per_rank.items()},
            "comm_mb": comm_bytes / 1024,
        }


def main() -> None:
    random.seed(42)
    eng = MoEEngine(num_experts=8, top_k=2, num_ranks=2)

    # Simulate 10 decode steps
    for step in range(10):
        batch = list(range(32))  # 32 tokens / step
        info = eng.step(batch)
        load = info["per_rank_load"]
        skew = max(load.values()) / max(1, min(load.values()))
        print(f"step {step}: per_rank={load}  load_skew={skew:.2f}  "
              f"comm={info['comm_mb']:.1f} MB")

    # Show that EP saves activation memory but not weight memory
    total_w = sum(e.weight_mb for e in eng.experts)
    per_rank_w = total_w / eng.num_ranks
    print(f"\ntotal expert weights (all ranks): {total_w/1024:.1f} GB")
    print(f"per-rank weights (if EP):         {per_rank_w/1024:.1f} GB")
    print("OK")


if __name__ == "__main__":
    main()
