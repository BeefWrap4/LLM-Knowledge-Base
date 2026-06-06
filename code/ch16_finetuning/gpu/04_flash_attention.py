# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.4.3 Flash Attention 使用
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch (>=2.0), flash-attn (optional)
# run: python 04_flash_attention.py --mock
# expected_runtime: <5s for mock / <30s for real
# expected_output: 展示 SDPA Flash kernel 路径 + 注意力计算正确性对比
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.4.3
# Interview hooks:
#   1. Flash Attention 为什么能将显存从 O(N^2) 降到 O(N)？分块 + Online Softmax 的关键？
#   2. 标准 Attention 矩阵 S = QK^T 的显存瓶颈在哪？为什么反向传播时需要重新计算？
#   3. PyTorch SDPA 是如何选择后端 kernel 的？enable_flash / enable_mem_efficient 区别？

"""
Flash Attention 使用 —— PyTorch 2.0+ SDPA / flash_attn 库
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_flash_attention_demo():
    """无 CUDA 环境下的概念演示"""
    print("[MOCK] Flash Attention 核心思路")
    print("  1. 分块（Tiling）:  Q/K/V 切成能在 SRAM 容纳的小块")
    print("  2. Online Softmax: 增量计算 softmax, 不存完整 N×N 矩阵")
    print("  3. 重计算（Recompute）: 反向时按需重算 attention 矩阵")
    print()
    print("[MOCK] PyTorch SDPA (>=2.0) 三种 kernel 路径")
    print("  - FLASH (Flash Attention 2)")
    print("  - EFFICIENT (Memory-Efficient, xformers 风格)")
    print("  - MATH (标准实现, 显存 O(N^2), 仅作 fallback)")
    print()
    print("[MOCK] 显存对比 (seq=4096, head=32, dim=128, bf16)")
    print("  Standard Attention: 4096×4096 × 2B × 32 heads ≈ 1 GB / layer")
    print("  Flash Attention:    O(N) ≈ 32 × 4096 × 128 × 2B ≈ 32 MB / layer")
    print("  -> 显存比 ~ 32x")
    print()
    print("OK")


def real_flash_attention_demo():
    """真实 Flash Attention 演示（需 CUDA）"""
    import torch
    import torch.nn.functional as F

    # 构造测试数据
    B, H, T, D = 2, 8, 1024, 64
    Q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, T, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, T, D, dtype=torch.bfloat16, device="cuda")

    # 方式1：PyTorch 原生 scaled_dot_product_attention
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    ):
        out_flash = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print(f"[SDPA-Flash] output shape: {tuple(out_flash.shape)}")

    # 方式2：数学实现（参考对照）
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False,
    ):
        out_math = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    diff = (out_flash - out_math).abs().max().item()
    print(f"[SDPA-Math]  max abs diff vs Flash: {diff:.4e}")

    # 方式3：flash_attn 库（若安装）
    try:
        from flash_attn import flash_attn_func
        # flash_attn 期望 (B, T, H, D) 布局
        Qf = Q.transpose(1, 2)
        Kf = K.transpose(1, 2)
        Vf = V.transpose(1, 2)
        out_fa = flash_attn_func(Qf, Kf, Vf, causal=True)
        out_fa = out_fa.transpose(1, 2)
        print(f"[flash_attn_func] output shape: {tuple(out_fa.shape)}")
    except ImportError:
        print("[flash_attn_func] 库未安装, 跳过 (pip install flash-attn)")

    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_flash_attention_demo()
    else:
        real_flash_attention_demo()
