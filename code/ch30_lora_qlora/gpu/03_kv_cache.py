# ---
# chapter: 40
# topic: 推理内存、量化与批处理
# topic_id: lora_qlora.kv_cache
# difficulty: ⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: none
# run: python 03_kv_cache.py
# expected_runtime: <1s
# expected_output: KV cache 内存对比表
# ---
# See: ../../../40_推理内存量化与批处理.md
#
# Interview hooks:
#   1. KV Cache 的显存公式？LLaMA-2-7B, seq=4096, bs=1 时约多少 GB？
#   2. MQA / GQA 如何按 query heads / KV heads 的比率减少 KV Cache？边界是什么？
#   3. KV Cache 与 Paged Attention 的关系？为什么连续分配会有碎片问题？
"""KV Cache 内存计算器 (纯计算, 无 GPU 加载).

KV Cache 显存公式:
  per_token = 2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes
  total     = per_token × seq_len × batch_size

GQA (Grouped Query Attention) 通过 num_kv_heads < num_heads 减少 KV cache.
相对同层数、head_dim、dtype 的 MHA，理论 K/V 元素数量比为 num_kv_heads / num_heads；
具体架构、滑动窗口、混合层与量化会改变总显存。
"""

from dataclasses import dataclass


@dataclass
class ModelKVConfig:
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_query_heads: int = 0  # 0 = 同 num_kv_heads (MHA)

    def __post_init__(self):
        if self.num_query_heads == 0:
            self.num_query_heads = self.num_kv_heads


# 主流模型 KV cache 配置 (来源: HuggingFace config.json)
CONFIGS = {
    "qwen2.5-0.5b": ModelKVConfig("Qwen2.5-0.5B", 24, 2, 64, num_query_heads=14),  # GQA
    "qwen2.5-7b": ModelKVConfig("Qwen2.5-7B", 28, 4, 128, num_query_heads=28),  # GQA
    "qwen2.5-72b": ModelKVConfig("Qwen2.5-72B", 80, 8, 128, num_query_heads=64),  # GQA
    "llama-3-8b": ModelKVConfig("Llama-3-8B", 32, 8, 128, num_query_heads=32),  # GQA
    "llama-2-7b": ModelKVConfig("Llama-2-7B", 32, 32, 128),  # MHA
}


def kv_per_token(cfg: ModelKVConfig, dtype_bytes: int = 2) -> int:
    """每个 token 的 KV cache 字节数 (K+V 加起来)."""
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * dtype_bytes


def total_kv(
    cfg: ModelKVConfig,
    max_seq_len: int = 4096,
    batch_size: int = 32,
    dtype_bytes: int = 2,
) -> int:
    """总 KV cache 字节数."""
    return kv_per_token(cfg, dtype_bytes) * max_seq_len * batch_size


def main():
    print("=== KV Cache 内存计算 (fp16, 2 bytes) ===\n")

    # 1) 不同模型单 token KV cache
    print("Per-token KV cache (单 token, fp16):")
    print(f"{'模型':<15} {'kv heads':>10} {'per tok':>12} {'gqa ratio':>12}")
    print("-" * 55)
    for cfg in CONFIGS.values():
        per_tok = kv_per_token(cfg) / 1024
        gqa_ratio = cfg.num_query_heads / cfg.num_kv_heads
        print(f"{cfg.name:<15} {cfg.num_kv_heads:>10} {per_tok:>9.1f}KB {gqa_ratio:>11.1f}x")

    # 2) 不同 ctx 长度 + batch 大小下的总 KV cache
    print("\n总 KV cache (4K ctx, batch=32, fp16):")
    print(f"{'模型':<15} {'total KV':>14} {'对比 Llama-2-7B':>20}")
    print("-" * 55)
    base_total = total_kv(CONFIGS["llama-2-7b"])
    for cfg in CONFIGS.values():
        total = total_kv(cfg) / (1024**3)
        ratio = total_kv(cfg) / base_total
        print(f"{cfg.name:<15} {total:>12.2f}GB {ratio:>19.2f}x")

    # 3) 长上下文场景 (32K)
    print("\n长上下文场景 (32K ctx, batch=1, fp16):")
    print(f"{'模型':<15} {'total KV':>14}")
    print("-" * 35)
    for cfg in CONFIGS.values():
        total = total_kv(cfg, max_seq_len=32768, batch_size=1) / (1024**3)
        print(f"{cfg.name:<15} {total:>12.2f}GB")

    # 4) 对比 FP16 vs INT8 vs INT4 (量化 KV cache)
    print("\n量化 KV cache 节省 (Qwen2.5-7B, 4K ctx, b=32):")
    print(f"{'dtype':<10} {'bytes':>8} {'total KV':>14} {'节省':>10}")
    print("-" * 50)
    cfg = CONFIGS["qwen2.5-7b"]
    fp16 = total_kv(cfg)
    for label, bytes_per_value in [("fp16", 2.0), ("int8", 1.0), ("int4", 0.5)]:
        # 这里只计算理想数据位宽；真实实现还会有 scale/zero-point 与对齐开销。
        v = fp16 * (bytes_per_value / 2.0)
        v_gb = v / (1024**3)
        saving = (1 - v / fp16) * 100
        print(f"{label:<10} {bytes_per_value:>8g} {v_gb:>12.2f}GB {saving:>9.0f}%")
    print("OK")


if __name__ == "__main__":
    main()
