# ---
# chapter: 25
# topic: KV Cache Memory Calculator (pure computation, no GPU)
# section: 25.1.2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: none
# run: python 05_kv_cache_memory_calculator.py
# expected_runtime: <1s
# expected_output: 估算 4 个真实模型在不同 batch/上下文下的 KV 显存占用
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.1.2
# Interview hooks:
#   1. KV cache 占显存的公式？哪些是 knobs？
#   2. GQA / MQA 如何减小 KV？(答: 减少 n_kv_heads；MLA 进一步压缩)
#   3. 为什么 decode 阶段是 memory-bound？(答: 每 token 读全量 KV，HBM 带宽瓶颈)
#   4. vLLM PagedAttention 对 KV 公式的影响？(答: 公式不变；block 切分不增总量)

"""KV Cache 内存计算器 (纯计算, 无需 GPU 加载).

公式:
    KV cache per token = 2 (K+V) × num_hidden_layers × num_kv_heads × head_dim × dtype_bytes
    Total KV cache    = per_token × max_seq_len × max_num_seqs × num_gpus (TP 分摊)

示例 (Qwen2.5-7B, GQA, n_kv_heads=4):
    - 2 × 28 layers × 4 kv_heads × 128 head_dim × 2 bytes (fp16) = 57,344 bytes/token
    - 57,344 × 32768 (32K context) × 64 (batch) ≈ 114 GB → 需 TP=2 / 4 在 24GB 显卡

示例 (Qwen2.5-0.5B, GQA, n_kv_heads=2):
    - 2 × 24 layers × 2 × 64 × 2 = 12,288 bytes/token
    - 12,288 × 4096 × 32 ≈ 1.5 GB → 单 16GB 显卡足以
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """LLM 架构关键参数 (GQA 友好)."""

    name: str
    num_hidden_layers: int
    num_attention_heads: int  # 实际 q heads (GQA 时常 > num_kv_heads)
    head_dim: int
    num_kv_heads: int  # GQA 共享 KV heads; MQA 时 = 1
    hidden_size: int


# 真实模型配置 (2026 常见, 数字来自 HF config.json)
KNOWN_MODELS: dict[str, ModelConfig] = {
    "qwen2.5-0.5b": ModelConfig(
        "Qwen2.5-0.5B",
        num_hidden_layers=24,
        num_attention_heads=14,
        head_dim=64,
        num_kv_heads=2,
        hidden_size=896,
    ),
    "qwen2.5-7b": ModelConfig(
        "Qwen2.5-7B",
        num_hidden_layers=28,
        num_attention_heads=28,
        head_dim=128,
        num_kv_heads=4,
        hidden_size=3584,
    ),
    "qwen2.5-72b": ModelConfig(
        "Qwen2.5-72B",
        num_hidden_layers=80,
        num_attention_heads=64,
        head_dim=128,
        num_kv_heads=8,
        hidden_size=8192,
    ),
    "llama-3.1-8b": ModelConfig(
        "Llama-3.1-8B",
        num_hidden_layers=32,
        num_attention_heads=32,
        head_dim=128,
        num_kv_heads=8,
        hidden_size=4096,
    ),
}


def kv_cache_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    """单个 token 的 KV cache 大小 (bytes).

    公式: 2 (K+V) × n_layers × n_kv_heads × head_dim × dtype_bytes
    """
    return 2 * cfg.num_hidden_layers * cfg.num_kv_heads * cfg.head_dim * dtype_bytes


def total_kv_cache(
    cfg: ModelConfig,
    max_seq_len: int = 4096,
    max_num_seqs: int = 32,
    dtype_bytes: int = 2,
    num_gpus: int = 1,
    tensor_parallel_size: int = 1,
) -> dict:
    """总 KV cache 大小 (bytes) + 详细分项.

    TP 分摊: 实际 per-GPU = total / (num_gpus × TP) (简化假设平均切).
    """
    per_token = kv_cache_per_token(cfg, dtype_bytes)
    per_seq = per_token * max_seq_len
    total = per_seq * max_num_seqs
    # TP 切 KV cache (n_kv_heads 维度均匀切)
    per_gpu = total // max(num_gpus * tensor_parallel_size, 1)

    return {
        "model": cfg.name,
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "per_seq_bytes": per_seq,
        "total_bytes": total,
        "total_gb": total / (1024**3),
        "per_gpu_bytes": per_gpu,
        "per_gpu_gb": per_gpu / (1024**3),
        "dtype_bytes": dtype_bytes,
        "max_seq_len": max_seq_len,
        "max_num_seqs": max_num_seqs,
    }


def recommend_gpu(per_gpu_gb: float, model_name: str = "") -> str:
    """根据 per-GPU KV cache 大小推荐硬件."""
    if per_gpu_gb < 8:
        return f"OK 8GB GPU (实际 {per_gpu_gb:.1f}GB)"
    elif per_gpu_gb < 16:
        return f"OK 16GB GPU ({per_gpu_gb:.1f}GB)"
    elif per_gpu_gb < 24:
        return f"OK 24GB GPU (RTX 4090/5090, {per_gpu_gb:.1f}GB)"
    elif per_gpu_gb < 40:
        return f"OK 40GB GPU (A100-40G, {per_gpu_gb:.1f}GB)"
    elif per_gpu_gb < 80:
        return f"OK 80GB GPU (A100-80G/H100, {per_gpu_gb:.1f}GB), 或 TP=2"
    else:
        return f"NEED 多卡 TP ({per_gpu_gb:.1f}GB), {model_name} 单卡放不下"


def main() -> None:
    print("=" * 70)
    print("KV Cache 内存计算器 (纯计算, 无 GPU 加载)")
    print("=" * 70)
    print()

    # 默认场景: 4K context, batch 32, fp16, 单卡
    print("[场景 1] 4K context × batch 32 × fp16, 单卡")
    print("-" * 70)
    for key, cfg in KNOWN_MODELS.items():
        info = total_kv_cache(cfg, max_seq_len=4096, max_num_seqs=32, dtype_bytes=2)
        print(f"\n{info['model']} ({key}):")
        print(f"  per token       = {info['per_token_kb']:.1f} KB")
        print(f"  total           = {info['total_gb']:.2f} GB")
        print(f"  per-GPU (单卡)  = {info['per_gpu_gb']:.2f} GB")
        print(f"  推荐            = {recommend_gpu(info['per_gpu_gb'], info['model'])}")

    print()
    print("=" * 70)
    print("[场景 2] Qwen2.5-7B 在 32K context × batch 64 (fp16/fp8/int8)")
    print("-" * 70)
    cfg = KNOWN_MODELS["qwen2.5-7b"]
    for dtype, label in [(2, "fp16"), (1, "fp8/int8")]:
        info = total_kv_cache(cfg, max_seq_len=32768, max_num_seqs=64, dtype_bytes=dtype)
        print(f"\n{label}:")
        print(f"  total           = {info['total_gb']:.1f} GB")
        print(f"  per-GPU (单卡)  = {info['per_gpu_gb']:.1f} GB  → {recommend_gpu(info['per_gpu_gb'])}")
        # TP=2/4 演示
        info_tp2 = total_kv_cache(
            cfg, max_seq_len=32768, max_num_seqs=64, dtype_bytes=dtype, tensor_parallel_size=2
        )
        print(
            f"  TP=2 per-GPU    = {info_tp2['per_gpu_gb']:.1f} GB  → {recommend_gpu(info_tp2['per_gpu_gb'])}"
        )
        info_tp4 = total_kv_cache(
            cfg, max_seq_len=32768, max_num_seqs=64, dtype_bytes=dtype, tensor_parallel_size=4
        )
        print(
            f"  TP=4 per-GPU    = {info_tp4['per_gpu_gb']:.1f} GB  → {recommend_gpu(info_tp4['per_gpu_gb'])}"
        )

    print()
    print("=" * 70)
    print("[场景 3] GQA 优势对比: Qwen2.5-7B (kv=4) vs 假设 MHA (kv=28)")
    print("-" * 70)
    cfg_gqa = KNOWN_MODELS["qwen2.5-7b"]
    cfg_mha = ModelConfig("Qwen2.5-7B-MHA (无GQA)", 28, 28, 128, 28, 3584)
    for label, c in [("GQA (kv=4)", cfg_gqa), ("MHA (kv=28)", cfg_mha)]:
        info = total_kv_cache(c, max_seq_len=4096, max_num_seqs=32, dtype_bytes=2)
        print(f"  {label:18s}  per-token = {info['per_token_kb']:6.1f} KB  total = {info['total_gb']:.2f} GB")
    # GQA 节省比例
    gqa_t = total_kv_cache(cfg_gqa, max_seq_len=4096, max_num_seqs=32, dtype_bytes=2)["total_gb"]
    mha_t = total_kv_cache(cfg_mha, max_seq_len=4096, max_num_seqs=32, dtype_bytes=2)["total_gb"]
    print(f"\n  GQA 节省: {100 * (1 - gqa_t / mha_t):.1f}% (相对 MHA)")


if __name__ == "__main__":
    main()
