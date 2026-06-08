# ---
# chapter: 25
# topic: Inference Engine Selection Decision Tree
# section: 25.7
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 12_engine_selection_decision_tree.py
# expected_runtime: <1s
# expected_output: 7 场景的引擎推荐
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.7
# Interview hooks:
#   1. vLLM vs TensorRT-LLM 核心权衡？(答: TRT 编译慢但运行极快; vLLM 启动快迭代友好)
#   2. 如何根据硬件预算选引擎？
#   3. 端侧 / 云端 推理引擎选择差异？
"""推理引擎选择决策函数 (纯逻辑, 无 GPU 加载).

输入: 硬件预算 (VRAM GB) + 延迟 SLO (ms) + 模型规模 + 部署场景
输出: 推荐引擎 + 关键理由 + 备选
"""

from dataclasses import dataclass
from enum import Enum


class Deployment(Enum):
    CLOUD = "cloud"
    EDGE_MAC = "edge_mac"
    EDGE_NVIDIA = "edge_nvidia"
    EDGE_CPU = "edge_cpu"
    SERVERLESS = "serverless"


@dataclass
class EngineRecommendation:
    engine: str
    config: dict
    reasoning: list
    alternatives: list


def pick_engine(
    model_size_b: float,
    vram_gb: float,
    latency_slo_ms: float,
    deployment: Deployment,
    needs_quantization: bool = True,
) -> EngineRecommendation:
    """根据硬件 + SLO + 部署 推荐推理引擎."""
    reasoning = []

    # 1. 端侧场景
    if deployment == Deployment.EDGE_MAC:
        return EngineRecommendation(
            engine="mlx_lm",
            config={"quant": "4bit", "batch_size": 1, "context_window": 4096},
            reasoning=["Apple Silicon: 选 MLX (统一内存, M-series 优化)"],
            alternatives=["llama.cpp (CPU fallback)"],
        )

    if deployment == Deployment.EDGE_NVIDIA:
        return EngineRecommendation(
            engine="tensorrt_llm",
            config={"quant": "int8", "engine_cache": True, "max_batch_size": 4},
            reasoning=["Jetson/Orin 边缘: TensorRT 编译后推理"],
            alternatives=["vllm (enforce_eager)"],
        )

    if deployment == Deployment.EDGE_CPU:
        return EngineRecommendation(
            engine="llama_cpp",
            config={"quant": "q4_k_m", "n_threads": 4, "n_ctx": 2048},
            reasoning=["CPU 推理: llama.cpp + GGUF 量化"],
            alternatives=["ctranslate2 (轻量任务)"],
        )

    if deployment == Deployment.SERVERLESS:
        return EngineRecommendation(
            engine="vllm",
            config={"quant": "awq" if needs_quantization else "fp16", "max_num_seqs": 32},
            reasoning=["Serverless 冷启动敏感: 选启动快的 vLLM (不用 TRT)"],
            alternatives=["tgi (HuggingFace TGI)"],
        )

    # 2. 云端场景 - 按 VRAM + latency SLO
    if vram_gb < 8:
        return EngineRecommendation(
            engine="vllm",
            config={"quant": "awq", "max_num_seqs": 4, "enforce_eager": True},
            reasoning=[f"VRAM {vram_gb}GB < 8GB: 需量化 (AWQ/GPTQ 4bit)"],
            alternatives=["tgi + bitsandbytes"],
        )

    if vram_gb < 16:
        reasoning.append(f"VRAM {vram_gb}GB (8-16): 单卡 7B 量化")
        if latency_slo_ms < 30:
            reasoning.append("严格 SLO < 30ms/tok: TensorRT 编译值得")
            return EngineRecommendation(
                engine="tensorrt_llm",
                config={"quant": "int8", "max_batch_size": 32, "tp_size": 1},
                reasoning=reasoning,
                alternatives=["vllm (快速迭代)"],
            )
        return EngineRecommendation(
            engine="vllm",
            config={"quant": "awq", "max_num_seqs": 16, "enforce_eager": True},
            reasoning=reasoning,
            alternatives=["tgi"],
        )

    if vram_gb < 24:
        return EngineRecommendation(
            engine="vllm",
            config={"quant": "fp16" if not needs_quantization else "awq", "max_num_seqs": 32},
            reasoning=[f"VRAM {vram_gb}GB (16-24): 单卡 7B fp16 或 13B 量化"],
            alternatives=["tensorrt_llm (极致优化)"],
        )

    if vram_gb < 80:
        reasoning.append(f"VRAM {vram_gb}GB (24-80): 单卡 70B 量化 或 7B-13B fp16")
        if model_size_b >= 70:
            return EngineRecommendation(
                engine="vllm",
                config={"quant": "awq", "max_num_seqs": 8, "tp_size": 1},
                reasoning=reasoning,
                alternatives=["tensorrt_llm", "sglang"],
            )
        return EngineRecommendation(
            engine="vllm",
            config={"quant": "fp16", "max_num_seqs": 64},
            reasoning=reasoning,
            alternatives=["tensorrt_llm"],
        )

    # VRAM >= 80: 多卡大模型
    reasoning.append(f"VRAM {vram_gb}GB (≥80): 多卡大模型, TensorRT 编译值得")
    if model_size_b >= 70:
        return EngineRecommendation(
            engine="tensorrt_llm",
            config={"quant": "fp8", "tp_size": 4, "pp_size": 1, "max_batch_size": 128},
            reasoning=reasoning,
            alternatives=["vllm (TP=4)"],
        )
    return EngineRecommendation(
        engine="vllm",
        config={"quant": "fp16", "tp_size": 2, "max_num_seqs": 128},
        reasoning=reasoning,
        alternatives=["tensorrt_llm", "sglang"],
    )


def main():
    print("=== 推理引擎选择决策树 ===\n")

    scenarios = [
        ("7B 量化 + 12GB 显卡 + 50ms SLO + 云端 (显存偏紧)", 7, 12, 50, Deployment.CLOUD),
        ("7B 量化 + 12GB 显卡 + 20ms SLO + 云端 (严格)", 7, 12, 20, Deployment.CLOUD),
        ("7B 量化 + 24GB 显卡 + 50ms SLO + 云端 (单卡)", 7, 24, 50, Deployment.CLOUD),
        ("70B fp8 + 80GB×4 + 30ms SLO + 云端", 70, 320, 30, Deployment.CLOUD),
        ("7B 4bit + MacBook M2 + 100ms SLO + 端侧", 7, 24, 100, Deployment.EDGE_MAC),
        ("7B int8 + Jetson Orin 16GB + 200ms SLO + 边缘", 7, 16, 200, Deployment.EDGE_NVIDIA),
        ("7B q4 + 无 GPU 笔记本 + 500ms SLO + 纯 CPU", 7, 0, 500, Deployment.EDGE_CPU),
        ("7B + Serverless 冷启动 + 50ms SLO", 7, 24, 50, Deployment.SERVERLESS),
    ]

    for desc, size, vram, slo, dep in scenarios:
        rec = pick_engine(size, vram, slo, dep)
        print(f"📊 {desc}")
        print(f"   引擎: {rec.engine}")
        print(f"   配置: {rec.config}")
        print(f"   理由: {'; '.join(rec.reasoning)}")
        if rec.alternatives:
            print(f"   备选: {rec.alternatives}")
        print()


if __name__ == "__main__":
    main()
