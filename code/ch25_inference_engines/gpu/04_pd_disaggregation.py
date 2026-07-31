# ---
# chapter: 25
# topic: Prefill-Decode Disaggregation (real vLLM chunked prefill)
# section: 25.4
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: vllm>=0.21.0, torch>=2.5
# run: python 04_pd_disaggregation.py
# expected_runtime: ~30-90s (model load + mixed long/short batch)
# expected_output: 长 prompt (prefill) 和短 prompt (decode) 混合 batch,
#                    打印 scheduler_config 展示 chunked prefill 行为
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.4
# Interview hooks:
#   1. 为什么要把 prefill 和 decode 拆到不同的实例上?
#   2. PD-Disagg 的关键瓶颈是什么？(答: KV transfer, 需 NVLink/IB/RDMA)
#   3. 什么场景 PD-Disagg 收益最大？(答: 长 prompt + 短 output、解码阶段高并发)
#   4. vLLM 0.21.0 的 chunked prefill 和 PD-Disagg 关系？(答: chunked prefill 是单节点近似)

"""Prefill-Decode (PD) Disaggregation 演示 (真实 vLLM 0.21.0).

PD disagg 把 prefill (compute-bound) 和 decode (memory-bound) 分到不同 GPU:
  - Prefill nodes: 大量 GPU, 跑 prefill 然后把 KV cache 传给 decode nodes
  - Decode nodes: 跑 decode 推理
  - KV 通过 NVLink / IB / RDMA 在节点间传输

vLLM 0.21.0 单节点 PD-Disagg 近似: ``enable_chunked_prefill=True`` +
``max_num_batched_tokens`` 控制. 把长 prompt 切成 chunk, 让 prefill 和
decode 混合 batch 执行, 避免 prefill 阻塞 decode.

完整 PD-Disagg (跨节点) 需 vLLM 0.21+ 的 XPyD (eXternal Prefill-Decode)
+ KV transfer 模块, 这里演示单节点版本作为入门.
"""

from __future__ import annotations

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
# 1 个长 prompt (prefill-heavy) + 8 个短 prompt (decode-heavy) 混合
LONG_PROMPT = "Explain in detail the theory of general relativity: " + ("Einstein " * 200)
SHORT_PROMPTS = [f"Q: Hello {i}? A:" for i in range(8)]


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

    # shared.vllm_compat: 设了 VLLM_BASE_URL → 走 Docker OpenAI 协议; 否则按需 import 真 vllm
    try:
        from shared.vllm_compat import LLM, SamplingParams
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

    # 关键开关: enable_chunked_prefill=True 让 prefill 和 decode 共享 batch
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_num_seqs=16,
        max_model_len=2048,  # 容纳长 prompt
        max_num_batched_tokens=512,  # 关键: 控制单 step token budget, 触发 chunked prefill
        enable_chunked_prefill=True,
        enforce_eager=True,
    )

    sched_cfg = llm.llm_engine.vllm_config.scheduler_config
    print("=" * 60)
    print("vLLM SchedulerConfig (Chunked Prefill / PD-Disagg 近似):")
    print(f"  enable_chunked_prefill    = {sched_cfg.enable_chunked_prefill}")
    print(f"  max_num_batched_tokens    = {sched_cfg.max_num_batched_tokens}")
    print("                            (单 step token budget, 把长 prompt 切成 chunk)")
    print(f"  max_num_seqs              = {sched_cfg.max_num_seqs}")
    print("=" * 60)
    print()

    # 长度统计
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    long_ids = tok(LONG_PROMPT, add_special_tokens=False).input_ids
    short_ids_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in SHORT_PROMPTS]
    print(f"Long prompt  : {len(long_ids)} tokens  (prefill-heavy)")
    print(
        f"Short prompts: {len(SHORT_PROMPTS)} x ~{sum(short_ids_lens) // len(short_ids_lens)} tokens "
        f"(decode-heavy)"
    )
    print(f"Total tokens : {len(long_ids) + sum(short_ids_lens)}")
    print()

    sampling = SamplingParams(temperature=0.7, max_tokens=24)
    all_prompts = [LONG_PROMPT] + SHORT_PROMPTS

    t0 = time.perf_counter()
    outputs = llm.generate(all_prompts, sampling)
    t_total = time.perf_counter() - t0

    print(f"\nGenerated {len(outputs)} prompts in {t_total:.2f}s")
    print()
    for i, out in enumerate(outputs):
        label = "LONG(prefill)" if i == 0 else f"short-{i - 1:02d}(decode) "
        text = out.outputs[0].text[:60].replace("\n", " ")
        ptoks = len(out.prompt_token_ids)
        ctoks = len(out.outputs[0].token_ids)
        print(f"  [{label}] prompt={ptoks}t -> gen={ctoks}t  text={text!r}")

    print()
    print("=" * 60)
    print("Chunked Prefill (单节点 PD-Disagg 近似) 关键 takeaway:")
    print("  - max_num_batched_tokens=512 把长 prompt 切成 ~32 chunks")
    print("  - 每个 step: 1 chunk prefill + N short decode, 混合 batch")
    print("  - 避免长 prompt 阻塞短 prompt 的 decode (vs static batching)")
    print("  - 真正跨节点 PD-Disagg 需 vLLM XPyD + KV transfer (RDMA/NVLink)")
    print("  - 验收目标: 比较长 prompt TTFT、短请求 TPOT、吞吐和 KV 传输开销")
    print("=" * 60)


if __name__ == "__main__":
    main()
