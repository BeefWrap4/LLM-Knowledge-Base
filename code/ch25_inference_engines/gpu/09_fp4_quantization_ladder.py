# ---
# chapter: 25
# topic: FP4 / NVFP4 / MXFP4 Quantization Ladder
# section: 25.5
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 09_fp4_quantization_ladder.py
# expected_runtime: <1s
# expected_output: 演示 FP32→FP16→FP8→FP4 的量化 ladder 及其误差/显存/吞吐取舍
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.5
# Interview hooks:
#   1. NVFP4 和 MXFP4 的核心区别？(答: NVFP4 全局共享 scale; MXFP4 块级 microscaling)
#   2. 4-bit 量化的精度损失如何控制？(答: 校准集、group size、outlier 处理)
#   3. PTQ vs QAT 的取舍？(答: PTQ 简单但损失大; QAT 慢但精度好)

"""Quantization ladder: FP32 → FP16/BF16 → FP8 → FP4.

Toy implementation that shows the *bit budget* and the *quantization
error* for a small weight tensor. In real engines the kernels are
fused into matmul (W4A16 / W4A8 GEMM).
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass


def fake_weights(n: int = 4096, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    # Mix of small and large magnitudes to test range
    return [rng.gauss(0, 0.05) + (0.5 if i % 100 == 0 else 0.0) for i in range(n)]


def quantize_per_tensor(w: list[float], levels: int) -> tuple[list[int], float, float]:
    """Symmetric per-tensor quantization. Returns (q_int, scale, zp)."""
    amax = max(abs(x) for x in w) or 1e-9
    scale = amax / ((levels - 1) / 2)
    q = [max(0, min(levels - 1, round((x / scale) + (levels - 1) / 2))) for x in w]
    return q, scale, (levels - 1) / 2


def dequantize(q: list[int], scale: float, zp: float) -> list[float]:
    return [(x - zp) * scale for x in q]


def rmse(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def main() -> None:
    w = fake_weights()
    print(f"weight tensor: n={len(w)}  fp32_bytes={len(w)*4}")

    ladder = [
        ("FP16/BF16",  2**16),
        ("INT8",       2**8),
        ("FP8 (E4M3)", 2**8),
        ("INT4",       2**4),
        ("FP4 (NVFP4)",2**4),
    ]
    print(f"\n{'format':<14}{'bits/w':>10}{'bytes(MB)':>12}{'RMSE':>10}")
    for name, levels in ladder:
        bits = math.log2(levels)
        size_mb = len(w) * bits / 8 / 1e6
        q, s, zp = quantize_per_tensor(w, levels)
        dq = dequantize(q, s, zp)
        err = rmse(w, dq)
        print(f"{name:<14}{bits:>10.1f}{size_mb:>12.3f}{err:>10.5f}")

    # Demonstrate microscaling-style (group-wise) which keeps FP4 quality
    print("\ngroup-wise FP4 (group_size=64) error vs per-tensor FP4:")
    group = 64
    errs = []
    for g in range(0, len(w), group):
        chunk = w[g:g + group]
        q, s, zp = quantize_per_tensor(chunk, 16)
        dq = dequantize(q, s, zp)
        errs.append(rmse(chunk, dq))
    grouped_rmse = sum(errs) / len(errs)
    print(f"  per-tensor FP4 RMSE: {rmse(w, dequantize(*quantize_per_tensor(w, 16))):.5f}")
    print(f"  group-wise FP4 RMSE: {grouped_rmse:.5f}  (usually 3-10x lower)")


if __name__ == "__main__":
    main()
