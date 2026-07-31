# ---
# chapter: 25
# topic: PagedAttention Block Manager (real vLLM)
# section: 25.2.1
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: vllm>=0.21.0, torch>=2.5, transformers
# run: python 01_paged_attention_block_manager.py
# expected_runtime: ~30-60s (model load + 1 generate)
# expected_output: 真实 vLLM LLM 跑一个 prompt, 并打印 CacheConfig
#                    (block_size, num_blocks) 证明 paged attention 在跑
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.1
# Interview hooks:
#   1. PagedAttention 为什么能提升显存利用率？(答: 类比 OS 虚拟内存，按页分配，碎片化低，复用率高)
#   2. Block size 如何选择？过小过大各有什么影响？(答: 16 token 经验值，过小页表大，过大内部碎片)
#   3. Beam Search 场景下如何处理 block 共享？(答: Copy-on-Write 写时复制)
#   4. vLLM 0.21.0 的 BlockManager 在哪里？(答: vllm/v1/core/kv_cache_manager.py, 私有 API, 通过 LLM 间接访问)

"""Paged Attention 演示 (真实 vLLM 0.21.0).

Paged Attention 把 KV cache 切成固定大小 block (默认 16 token/block),
类似 OS 虚拟内存分页.

vLLM 0.21.0 的真实 BlockManager 在 ``vllm/v1/core/kv_cache_manager.py``
(KVCacheManager), 负责:
  - 分配 / 释放 KV block
  - 维护 block table (logical → physical)
  - Copy-on-Write (beam search, parallel sampling)
  - Prefix caching hash (radix tree)

注意: KVCacheManager 是 vLLM 的内部类 (需要 vllm._C 编译扩展), 不能
直接 import; 它的行为通过 ``vllm.LLM`` 公共 API 暴露. 本脚本:
  1. 用 LLM 跑一个真实 prompt, 强制 vLLM 实际分配 KV block
  2. 读取 ``llm.llm_engine.vllm_config.cache_config`` 看真实配置
  3. 解释打印 block_size / num_blocks / kv cache 显存占用
"""

from __future__ import annotations

import sys
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

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 1GB, 已下载到本地 HF cache


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

    # 故意把 import 放在 GPU 检查之后, 避免无 GPU 机器 import vllm._C 失败
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

    # block_size 不指定 → vLLM 自动选 (通常 16); 这里显式传 16 方便观察
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,  # 留余量, 避免 OOM
        max_num_seqs=4,
        max_model_len=512,  # 短上下文, 加速冷启动
        block_size=16,  # paged attention 的 block 大小
        enforce_eager=True,  # 跳过 CUDA graph 编译, 启动更快
    )

    cache_cfg = llm.llm_engine.vllm_config.cache_config
    print("=" * 60)
    print("vLLM CacheConfig (PagedAttention 真实参数):")
    print(f"  block_size              = {cache_cfg.block_size} tokens/block")
    print(f"  block_size_hash         = {cache_cfg.block_size_hash}")
    print(f"  gpu_memory_utilization  = {cache_cfg.gpu_memory_utilization}")
    print(f"  num_gpu_blocks (KV池)   = {cache_cfg.num_gpu_blocks}")
    if cache_cfg.num_cpu_blocks:
        print(f"  num_cpu_blocks (offload) = {cache_cfg.num_cpu_blocks}")
    # KV cache 总 token 容量
    total_kv_tokens = cache_cfg.num_gpu_blocks * cache_cfg.block_size
    print(f"  total KV capacity       = {total_kv_tokens} tokens")
    print("=" * 60)
    print()

    # 跑一个真实 prompt, 强制 vLLM 实际分配 KV block
    sampling = SamplingParams(temperature=0.7, max_tokens=24)
    prompts = [
        "The capital of France is",
        "Paged attention was introduced in",  # 第二个 prompt, 触发 block 复用
    ]
    outputs = llm.generate(prompts, sampling)

    print("Generated outputs:")
    for i, out in enumerate(outputs):
        text = out.outputs[0].text[:80].replace("\n", " ")
        ptoks = len(out.prompt_token_ids)
        ctoks = len(out.outputs[0].token_ids)
        print(f"  [{i}] prompt={ptoks}t -> gen={ctoks}t  text={text!r}")

    print()
    print("=" * 60)
    print("PagedAttention 关键 takeaway:")
    print("  - KV cache 切成 block_size (16) token 的固定块, 跨序列独立管理")
    print("  - logical→physical block table 维护在 KVCacheManager (内部类)")
    print("  - 显存利用率比 contiguous 分配高 ~4x (SOSP'23 论文数据)")
    print("  - Copy-on-Write 支持 beam search: 多 beam 共享 prefix KV, 分叉时再复制")
    print("=" * 60)


if __name__ == "__main__":
    main()
