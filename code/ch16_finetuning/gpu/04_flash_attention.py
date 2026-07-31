# ---
# chapter: 16
# topic: Flash Attention (torch SDPA 真实 benchmark)
# section: 16.4.3
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch>=2.0
# run: python 04_flash_attention.py
# expected_runtime: 30-60s
# expected_output: 标准 vs Flash Attention 性能对比
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.4.3
#
# Interview hooks:
#   1. Flash Attention 为什么能将显存从 O(N^2) 降到 O(N)？分块 + Online Softmax 的关键？
#   2. 标准 Attention 矩阵 S = QK^T 的显存瓶颈在哪？为什么反向传播时需要重新计算？
#   3. PyTorch SDPA 是如何选择后端 kernel 的？enable_flash / enable_mem_efficient 区别？
"""Flash Attention 演示 (torch SDPA).

PyTorch 2.0+ 通过 torch.nn.functional.scaled_dot_product_attention
自动选择最优 attention 后端 (Flash Attention 2, Memory-Efficient, Math).

此处对比:
  - naive: 标准 O(N^2) attention, 显存 O(N^2)
  - SDPA:  自动选择 Flash/MemEfficient 后端
"""

import sys
import time
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch
import torch.nn.functional as F

from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


def naive_attention(q, k, v):
    """标准 causal O(N^2) attention — 显式 softmax, 完整 attn 矩阵."""
    scale = q.size(-1) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    causal_mask = torch.ones(
        q.size(-2),
        k.size(-2),
        dtype=torch.bool,
        device=q.device,
    ).triu(diagonal=1)
    attn = attn.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return attn @ v


def sdpa_attention(q, k, v):
    """PyTorch SDPA (自动 Flash Attention 后端)."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def benchmark(fn, q, k, v, n_warmup=3, n_runs=10):
    """测平均延迟 (ms)."""
    for _ in range(n_warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = fn(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000 / n_runs
    return out, elapsed


def main():
    if skip_if_mock("an NVIDIA GPU with a supported PyTorch SDPA backend"):
        return
    check_hardware()

    # 测试配置: batch=2, heads=8, seq=2048, head_dim=64
    B, H, S, D = 2, 8, 2048, 64
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print("=== Flash Attention (torch SDPA) vs 标准 attention ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}, dtype=bf16")
    print(f"   attn matrix 大小: {H} × {S} × {S} × 2B = {H * S * S * 2 / 1024:.1f}KB/层")
    print()

    # 1) Naive attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out_naive, naive_ms = benchmark(naive_attention, q, k, v)
    naive_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # 2) SDPA (Flash / MemEfficient 自动选择)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out_sdpa, sdpa_ms = benchmark(sdpa_attention, q, k, v)
    sdpa_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # 3) 数值正确性 (loss 容忍)
    max_diff = (out_naive - out_sdpa).abs().max().item()
    rms_diff = (out_naive.float() - out_sdpa.float()).square().mean().sqrt().item()
    torch.testing.assert_close(
        out_naive.float(),
        out_sdpa.float(),
        atol=5e-2,
        rtol=5e-2,
    )

    print("naive attention:")
    print(f"  latency:  {naive_ms:.2f}ms")
    print(f"  VRAM:     {naive_vram * 1024:.1f}MB")
    print()
    print("SDPA (Flash):")
    print(f"  latency:  {sdpa_ms:.2f}ms")
    print(f"  VRAM:     {sdpa_vram * 1024:.1f}MB")
    print()
    print(f"数值差异: max abs = {max_diff:.4e}, RMS = {rms_diff:.4e}")
    print()

    speedup = naive_ms / sdpa_ms
    print(f"speedup: {speedup:.2f}x | VRAM 节省: {(1 - sdpa_vram / naive_vram) * 100:.0f}%")
    print()
    print("✅ SDPA 自动选择最优后端 (Flash Attention 2 / MemEfficient / Math)")
    print("   在长序列 (S≥4096) 时 Flash 优势更显著 (O(N²) → O(N) 显存)")
    print("OK")


if __name__ == "__main__":
    main()
