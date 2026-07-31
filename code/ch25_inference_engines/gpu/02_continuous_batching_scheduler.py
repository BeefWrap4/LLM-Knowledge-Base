# ---
# chapter: 25
# topic: Continuous Batching Scheduler (real vLLM AsyncLLMEngine)
# section: 25.2.1 / 25.4
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: vllm>=0.21.0, torch>=2.5
# run: python 02_continuous_batching_scheduler.py
# expected_runtime: ~30-60s (model load + 8 concurrent generates)
# expected_output: 8 个并发请求通过 AsyncLLMEngine 跑, 打印每个 req 的 first/last token 时延
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.1
# Interview hooks:
#   1. 什么是 continuous batching? 和 static batching 比有什么优势?
#   2. Iteration-level scheduling 解决了什么问题？(答: padding 浪费、head-of-line blocking)
#   3. vLLM 如何在每个 decode step 决定插入/驱逐哪些请求?
#   4. vLLM 0.21.0 的 scheduler 在哪里？(答: vllm/v1/core/scheduler.py, 私有 API)

"""Continuous Batching 演示 (真实 vLLM 0.21.0 AsyncLLMEngine).

Continuous batching 允许 decode 阶段动态插入/驱逐请求, 不像 static batching
要等所有请求完成才能插入新请求.

vLLM 的连续批处理通过以下机制实现 (vllm/v1/core/scheduler.py):
  - 每个 decode step 重新选择 running batch: 完成的 evict, 等待的 admit
  - Iteration-level scheduling；收益必须在相同模型、流量与延迟 SLO 下对照测试
  - PagedAttention 让变长 batch 不需要 contiguous KV

注意: vLLM 的 scheduler 是内部类 (需 vllm._C), 行为通过 ``AsyncLLMEngine.generate``
async 迭代器暴露. 多个并发任务 = 多个并发请求被自动批处理.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import (
    gpu_summary,
    require_nvidia_gpu,
    skip_if_mock,
    skip_unless_enabled,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_CONCURRENT = 8
PROMPTS = [
    "Q: What is 2+2? A:",
    "Q: Name a planet. A:",
    "Q: Capital of Japan? A:",
    "Q: Largest ocean? A:",
    "Q: Who wrote Hamlet? A:",
    "Q: Speed of light? A:",
    "Q: Boiling point of water in C? A:",
    "Q: H2O is? A:",
]


async def stream_one(engine, prompt: str, req_id: str, sampling) -> dict:
    """Run a single request via AsyncLLMEngine, return timing + output."""
    t0 = time.perf_counter()
    first_t = None
    last_text = ""
    n_tokens = 0
    async for out in engine.generate(prompt, sampling, req_id):
        if first_t is None:
            first_t = time.perf_counter()
        last_text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        if out.finished:
            break
    t_end = time.perf_counter()
    return {
        "req_id": req_id,
        "ttft_ms": (first_t - t0) * 1000 if first_t else 0,
        "total_ms": (t_end - t0) * 1000,
        "n_tokens": n_tokens,
        "text": last_text[:60].replace("\n", " "),
    }


async def run() -> None:
    # 故意把 import 放在 async 函数内, 避免顶层 import vllm._C
    # shared.vllm_compat: 设了 VLLM_BASE_URL → 走 Docker OpenAI 协议; 否则按需 import 真 vllm
    try:
        from shared.vllm_compat import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    except ModuleNotFoundError as e:
        if "vllm._C" in str(e):
            print("=" * 60)
            print("ERROR: vllm._C compiled extension is missing.")
            print("  vLLM 0.21.0 在 Windows 上需要从源码编译 C++/CUDA 扩展,")
            print("  当前的 pip install 缺 vllm._C.pyd.")
            print()
            print("修复方案 (任选其一):")
            print("  1. Linux 机器: pip install vllm==0.21.0  (官方支持)")
            print("  2. WSL2 + CUDA: 在 WSL2 里 pip install vllm")
            print("  3. Docker:  vllm/vllm-openai:0.21.0 镜像")
            print()
            print("本脚本代码正确, 在 vllm._C 可用的环境可直接跑通.")
            print("=" * 60)
            return
        raise

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.5,
            max_num_seqs=N_CONCURRENT,
            max_model_len=512,
            enforce_eager=True,
            disable_log_stats=True,  # 关闭 per-step log, 输出更清晰
        )
    )

    sampling = SamplingParams(temperature=0.7, max_tokens=24)
    sched_cfg = engine.llm_engine.vllm_config.scheduler_config

    print("=" * 60)
    print("vLLM SchedulerConfig (Continuous Batching 真实参数):")
    print(f"  max_num_seqs        = {sched_cfg.max_num_seqs}  (concurrent batch 上限)")
    print(f"  max_num_batched_tokens = {sched_cfg.max_num_batched_tokens}  (per-step token budget)")
    print(f"  enable_chunked_prefill = {sched_cfg.enable_chunked_prefill}")
    print("=" * 60)
    print(f"\nLaunching {N_CONCURRENT} concurrent requests...")

    t_start = time.perf_counter()
    tasks = [stream_one(engine, p, f"r{i}", sampling) for i, p in enumerate(PROMPTS)]
    results = await asyncio.gather(*tasks)
    t_total = time.perf_counter() - t_start

    print(f"\nAll {N_CONCURRENT} requests done in {t_total:.2f}s")
    print()
    print(f"{'req':<6} {'ttft(ms)':<10} {'total(ms)':<11} {'tokens':<8} text")
    print("-" * 80)
    for r in results:
        print(
            f"{r['req_id']:<6} {r['ttft_ms']:<10.1f} {r['total_ms']:<11.1f} {r['n_tokens']:<8} {r['text']!r}"
        )
    print()
    print("=" * 60)
    print("Continuous Batching 关键 takeaway:")
    print("  - 8 个 req 同时发, vLLM scheduler 每个 decode step 重新挑 batch")
    print("  - 短 req (Q+A) 完成后立即被 evict, 不阻塞其他 req (vs static batching)")
    print("  - TTFT 反映 prefill 时延; total 反映 continuous 调度效率")
    print("  - 真实 vLLM scheduler: vllm/v1/core/scheduler.py (内部, 不能直接 import)")
    print("=" * 60)


def main() -> None:
    if skip_if_mock("an NVIDIA GPU, CUDA, vLLM, and local model weights"):
        return
    if skip_unless_enabled(
        "VLLM_EXAMPLE_RUN", "the Linux/WSL2 vLLM runtime and local model weights"
    ):
        return
    require_nvidia_gpu(min_vram_gb=8)
    print(gpu_summary())
    print()
    asyncio.run(run())


if __name__ == "__main__":
    main()
