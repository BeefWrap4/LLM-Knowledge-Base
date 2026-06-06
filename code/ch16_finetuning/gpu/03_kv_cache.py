# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.4.2 KV Cache 简化实现
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 03_kv_cache.py --mock
# expected_runtime: <5s for mock / <10s for real
# expected_output: KV Cache 显存估算 + 顺序 update/get 演示
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.4.2
# Interview hooks:
#   1. KV Cache 的显存公式？LLaMA-2-7B, seq=4096, bs=1 时约多少 GB？
#   2. MQA / GQA（Multi/Grouped Query Attention）如何将 KV Cache 压缩到 1/4~1/8？
#   3. KV Cache 与 Paged Attention 的关系？为什么连续分配会有碎片问题？

"""
KV Cache 概念实现 —— 缓存每层的 K/V 避免自回归重复计算
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_kv_cache_demo():
    """无 CUDA 环境下的演示 + 显存估算"""
    # LLaMA-2-7B 典型配置
    B, L, H, T, D, dtype_bytes = 1, 32, 32, 4096, 128, 2  # bf16
    kv_bytes = 2 * B * L * H * T * D * dtype_bytes
    kv_gb = kv_bytes / (1024 ** 3)
    print(f"[MOCK] LLaMA-2-7B KV Cache 显存估算")
    print(f"  batch={B}, layers={L}, heads={H}, seq={T}, head_dim={D}, bf16")
    print(f"  公式: 2 × B × L × H × T × D × bytes")
    print(f"  = 2 × {B} × {L} × {H} × {T} × {D} × {dtype_bytes}")
    print(f"  = {kv_bytes/1e9:.2f} GB")
    print()
    print("  -> 配合 MQA (num_kv_heads=1) 或 GQA (num_kv_heads=8)")
    print("     可将 KV Cache 压缩到 1/32 ~ 1/4（视模型而定）")
    print()
    print("[MOCK] KVCache 简化接口（伪代码）")
    print("  __init__(num_layers, batch_size, num_heads, head_dim, max_seq_len)")
    print("  get(layer_idx) -> (K, V) of shape (B, H, T_cur, D)")
    print("  update(layer_idx, new_k, new_v)  # 追加到 current_len 之后")
    print("  increment(delta=1)               # current_len += 1")
    print()
    print("OK")


def real_kv_cache_demo():
    """真实 KV Cache 演示（需 CUDA）"""
    import torch

    class KVCache:
        """KV Cache 简化实现"""

        def __init__(self, num_layers, batch_size, num_heads, head_dim, max_seq_len):
            self.num_layers = num_layers
            # 预分配 [layers, 2(k/v), batch, heads, max_seq, head_dim]
            self.cache = torch.zeros(
                num_layers, 2, batch_size, num_heads, max_seq_len, head_dim,
                dtype=torch.bfloat16, device="cuda",
            )
            self.current_len = 0

        def get(self, layer_idx):
            """获取指定层的 K/V（到当前长度）"""
            k = self.cache[layer_idx, 0, :, :, :self.current_len, :]
            v = self.cache[layer_idx, 1, :, :, :self.current_len, :]
            return k, v

        def update(self, layer_idx, new_k, new_v):
            """追加新的 K/V"""
            seq_len = new_k.shape[2]
            end = self.current_len + seq_len
            self.cache[layer_idx, 0, :, :, self.current_len:end, :] = new_k
            self.cache[layer_idx, 1, :, :, self.current_len:end, :] = new_v

        def increment(self, delta=1):
            self.current_len += delta

    # 演示：32 层 7B 模型的 KV Cache
    cache = KVCache(
        num_layers=32,
        batch_size=1,
        num_heads=32,
        head_dim=128,
        max_seq_len=512,
    )

    print(f"缓存形状: {tuple(cache.cache.shape)}")
    print(f"占用显存: {cache.cache.numel() * cache.cache.element_size() / 1024**2:.2f} MB")

    # 模拟 step 1: 预填充 4 个 token
    fake_k = torch.randn(1, 32, 4, 128, dtype=torch.bfloat16, device="cuda")
    fake_v = torch.randn(1, 32, 4, 128, dtype=torch.bfloat16, device="cuda")
    for layer in range(32):
        cache.update(layer, fake_k[layer:layer+1], fake_v[layer:layer+1])
    cache.increment(4)
    print(f"After prefill: current_len={cache.current_len}")
    k0, v0 = cache.get(0)
    print(f"Layer 0 K shape={tuple(k0.shape)}")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_kv_cache_demo()
    else:
        real_kv_cache_demo()
