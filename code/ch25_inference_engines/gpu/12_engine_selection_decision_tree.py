# ---
# chapter: 25
# topic: Engine Selection Decision Tree
# section: 25.7
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 12_engine_selection_decision_tree.py
# expected_runtime: <1s
# expected_output: 给定硬件 / 场景，输出推荐的推理引擎
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.7
# Interview hooks:
#   1. 选 vLLM 还是 SGLang？(答: 高 QPS + 共享前缀 → SGLang/RadixAttention; 通用 → vLLM)
#   2. 选 vLLM 还是 TensorRT-LLM？(答: 极致性能 + 编译可接受 → TRT-LLM; 迭代快 → vLLM)
#   3. 何时用 llama.cpp？(答: CPU/Apple Silicon/嵌入式, 无 GPU)

"""Inference engine selection tree (2026 edition)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class UseCase:
    hardware: str          # "H100" | "A100" | "L40S" | "RTX4090" | "AppleSilicon" | "CPU"
    workload: str          # "chat" | "rag" | "code" | "agent" | "batch-eval"
    concurrency: int       # peak QPS
    context_len: int       # typical prompt+output tokens
    shared_prefix: bool    # heavy system prompts / few-shot?
    compile_budget_min: int  # how long can you wait for an engine build?
    needs_moe: bool


def recommend(uc: UseCase) -> tuple[str, str]:
    """Return (engine, reason). The decision tree is intentionally simple."""
    # 1) No-GPU path
    if uc.hardware in ("AppleSilicon", "CPU"):
        return "llama.cpp (GGUF, Metal/AVX)", "no discrete GPU; llama.cpp is mature"

    # 2) MoE serving on Hopper/Blackwell
    if uc.needs_moe:
        if uc.hardware in ("H100", "H200", "B200"):
            return "SGLang (DeepEP, expert parallel)", "best MoE EP support on Hopper/Blackwell"
        return "vLLM (EP via DeepSeek-style all-to-all)", "stable MoE serving"

    # 3) Shared prefix workloads (RAG, agent, system prompt + tools)
    if uc.shared_prefix and uc.concurrency > 50:
        return "SGLang (RadixAttention + PD-Disagg)", "high prefix-reuse + high QPS"

    # 4) Hard latency SLO on NVIDIA
    if uc.compile_budget_min >= 30 and uc.hardware.startswith(("H", "L", "A")):
        return "TensorRT-LLM", "compile time OK; needs peak tok/s on NVIDIA"

    # 5) Default: vLLM
    return "vLLM (PagedAttention + continuous batching)", \
        "best general-purpose LLM server; huge model coverage"


def main() -> None:
    cases = [
        UseCase("H100",  "chat",    200,  4096, True,  20, False),
        UseCase("A100",  "rag",     500,  8192, True,  10, False),
        UseCase("H100",  "code",     20,  16384, False, 60, False),
        UseCase("H200",  "chat",    300,  4096, False, 5,  True),
        UseCase("AppleSilicon", "chat", 5, 4096, False, 0, False),
        UseCase("RTX4090", "batch-eval", 1, 2048, False, 5, False),
        UseCase("H100",  "agent",   150,  16384, True,  15, False),
    ]
    print(f"{'HW':<14}{'workload':<12}{'QPS':>6}{'moe':>5}  ->  engine")
    print("-" * 78)
    for uc in cases:
        eng, why = recommend(uc)
        print(f"{uc.hardware:<14}{uc.workload:<12}{uc.concurrency:>6}"
              f"{'Y' if uc.needs_moe else 'N':>5}  ->  {eng}")
        print(f"{'':>32}  reason: {why}")
    print("OK")


if __name__ == "__main__":
    main()
