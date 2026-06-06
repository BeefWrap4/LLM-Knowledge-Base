# ---
# chapter: 25
# topic: TensorRT-LLM Engine Build (Mock)
# section: 25.2.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 08_tensorrt_llm_build_mock.py
# expected_runtime: <1s
# expected_output: 模拟 TensorRT-LLM build 流程：graph capture → kernel autotune → engine serialize
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.3
# Interview hooks:
#   1. TensorRT-LLM 和 vLLM 的核心权衡？(答: TRT 编译慢但运行极快；vLLM 启动快，迭代友好)
#   2. In-flight batching 是什么？(答: decode 阶段动态插入/驱逐，类似 continuous batching)
#   3. kernel autotune 在 build 时做什么？(答: 在目标硬件上选最快的 GEMM/attention kernel)

"""Mock the TensorRT-LLM build + serve pipeline.

In production:
    trtllm-build --checkpoint_dir ... --output_dir engine \
                 --gemm_plugin fp8 --max_batch_size 64 --max_seq_len 8192
    trtllm-serve engine/

This file mirrors the *stages* so you can talk about them in interviews.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class BuildConfig:
    max_batch_size: int = 64
    max_seq_len: int = 8192
    gemm_plugin: str = "fp8"   # fp16 | bf16 | fp8 | fp4
    attention_plugin: str = "trtllm"
    tp_size: int = 1
    pp_size: int = 1
    enable_in_flight_batching: bool = True


@dataclass
class Engine:
    name: str
    config: BuildConfig
    artifacts: list[str] = field(default_factory=list)
    build_seconds: float = 0.0


def build_engine(cfg: BuildConfig, model_id: str = "llama-3-8b") -> Engine:
    eng = Engine(name=f"{model_id}-trt", config=cfg)
    t0 = time.time()

    # Stage 1: parse + graph capture
    print(f"[1/4] graph capture (TP={cfg.tp_size} PP={cfg.pp_size})")
    eng.artifacts.append("graph.txt")

    # Stage 2: kernel autotune — pick best GEMM kernel for this shape
    print(f"[2/4] autotune kernels (gemm={cfg.gemm_plugin}, attn={cfg.attention_plugin})")
    candidates = ["cublas", "cutlass", "trtllm-fp8", "trtllm-fp4"]
    eng.artifacts.append(f"selected: {candidates[2] if 'fp8' in cfg.gemm_plugin else candidates[3]}")

    # Stage 3: build engine plan
    print("[3/4] build engine.plan")
    eng.artifacts.append("engine.plan")

    # Stage 4: serialize + checksum
    print("[4/4] serialize")
    eng.artifacts.append("engine.cache")

    eng.build_seconds = round(time.time() - t0 + 600.0, 1)  # mock 10-min build
    return eng


def serve(eng: Engine, prompt_len: int, out_len: int) -> dict:
    """Mock request: would dispatch to executor with in-flight batching."""
    if not eng.config.enable_in_flight_batching:
        raise RuntimeError("in-flight batching must be enabled in 2026")
    return {
        "engine": eng.name,
        "ttft_ms_est": round(20 + prompt_len * 0.05, 1),
        "tpot_ms_est": round(8.0 if "fp8" in eng.config.gemm_plugin else 12.0, 1),
        "throughput_tokens_per_s_per_gpu_est": 4200,
    }


def main() -> None:
    cfg = BuildConfig(max_batch_size=128, max_seq_len=16384,
                      gemm_plugin="fp8", tp_size=2, pp_size=1)
    eng = build_engine(cfg, model_id="llama-3-70b")
    print(f"\nbuild complete in {eng.build_seconds}s")
    print("artifacts:", eng.artifacts)

    resp = serve(eng, prompt_len=2048, out_len=256)
    print("mock serve response:", resp)
    print("OK")


if __name__ == "__main__":
    main()
