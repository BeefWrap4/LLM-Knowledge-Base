# ---
# chapter: 25
# topic: KV Cache Memory Calculator
# section: 25.1.2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 05_kv_cache_memory_calculator.py
# expected_runtime: <1s
# expected_output: 估算 LLaMA-70B 在不同 batch / 上下文下的 KV 显存占用
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.1.2
# Interview hooks:
#   1. KV cache 占显存的公式？哪些是 knobs？
#   2. GQA / MQA 如何减小 KV？(答: 减少 n_kv_heads；MLA 进一步压缩)
#   3. 为什么 decode 阶段是 memory-bound？(答: 每 token 读全量 KV，HBM 带宽瓶颈)

"""KV cache memory estimator for transformer inference.

Formula (per layer, per token, in bytes):
    KV = 2 * n_kv_heads * head_dim * precision_bytes

Total per request:
    KV_total = n_layers * seq_len * KV_per_token
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int


PRESETS: dict[str, ModelSpec] = {
    "llama-3-8b":    ModelSpec("LLaMA-3-8B",   n_layers=32,  n_kv_heads=8,  head_dim=128),
    "llama-3-70b":   ModelSpec("LLaMA-3-70B",  n_layers=80,  n_kv_heads=8,  head_dim=128),
    "llama-3.1-405b":ModelSpec("LLaMA-3.1-405B",n_layers=126, n_kv_heads=8,  head_dim=128),
    "qwen2-72b":     ModelSpec("Qwen2-72B",    n_layers=80,  n_kv_heads=8,  head_dim=128),
    "mistral-7b":    ModelSpec("Mistral-7B",   n_layers=32,  n_kv_heads=8,  head_dim=128),
    "mixtral-8x7b":  ModelSpec("Mixtral-8x7B", n_layers=32,  n_kv_heads=8,  head_dim=128),
    "deepseek-v3":   ModelSpec("DeepSeek-V3",  n_layers=61,  n_kv_heads=128,head_dim=128),
}


def precision_bytes(name: str) -> int:
    return {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1, "fp4": 0.5}[name]


def kv_bytes_per_token(spec: ModelSpec, precision: str = "fp16") -> float:
    """Per layer, per token, KV cache bytes (K and V combined)."""
    return 2.0 * spec.n_kv_heads * spec.head_dim * precision_bytes(precision)


def total_kv_gb(spec: ModelSpec, seq_len: int, batch: int, precision: str = "fp16") -> float:
    per_token = kv_bytes_per_token(spec, precision)
    total = spec.n_layers * seq_len * per_token * batch
    return total / (1024 ** 3)


def main() -> None:
    print(f"{'model':<20}{'seq':>8}{'batch':>8}{'fp16 GB':>10}{'fp8 GB':>10}{'fp4 GB':>10}")
    scenarios = [
        ("llama-3-8b",  4096,  16),
        ("llama-3-8b",  4096,  64),
        ("llama-3-70b", 8192,  16),
        ("llama-3-70b", 8192,  32),
        ("llama-3-70b", 8192, 128),
        ("llama-3.1-405b", 8192, 32),
        ("deepseek-v3", 8192, 32),
    ]
    for name, seq, batch in scenarios:
        spec = PRESETS[name]
        fp16 = total_kv_gb(spec, seq, batch, "fp16")
        fp8  = total_kv_gb(spec, seq, batch, "fp8")
        fp4  = total_kv_gb(spec, seq, batch, "fp4")
        print(f"{name:<20}{seq:>8}{batch:>8}{fp16:>10.2f}{fp8:>10.2f}{fp4:>10.2f}")

    # 70B @ 8K ctx, batch 32, fp16 — direct calculation
    #   2 * 80 layers * 8192 seq * 8 heads * 128 dim * 2 bytes * 32 batch
    #   = 80.0 GB  (KV-only; weights themselves are ~140 GB → this is why
    #     multi-70B serving needs 4×/8× H100s with TP/PP)
    s = PRESETS["llama-3-70b"]
    check = total_kv_gb(s, seq_len=8192, batch=32, precision="fp16")
    assert 79.5 < check < 80.5, f"sanity check failed: {check}"
    print(f"\nsanity: LLaMA-70B 8K×32 fp16 = {check:.2f} GB  (KV-only; weights ~140 GB)")
    print("OK")


if __name__ == "__main__":
    main()
