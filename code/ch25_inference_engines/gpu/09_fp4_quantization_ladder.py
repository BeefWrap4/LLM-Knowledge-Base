# ---
# chapter: 25
# topic: FP4 / INT4 / INT8 量化阶梯 (真实 bitsandbytes)
# section: 25.5
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: transformers, bitsandbytes, accelerate
# run: python 09_fp4_quantization_ladder.py
# expected_runtime: 60-180s (3 quantization types, model reload)
# expected_output: fp16/int8/fp4 三种量化的 VRAM + 延迟对比
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.5
# Interview hooks:
#   1. NF4 vs FP4 区别？(答: NF4 (4-bit NormalFloat) 数据类型专为正态分布权重设计; FP4 真浮点)
#   2. 4-bit 量化精度损失如何控制？(答: 校准集 / group size / double quant / outlier 处理)
#   3. PTQ vs QAT 取舍？(答: PTQ 简单但损失大; QAT 慢但精度好)
"""FP16 / INT8 / NF4 量化阶梯 (真实 bitsandbytes + transformers).

脚本在同一机器、模型、提示词和运行参数下测量峰值显存与一次生成延迟。
结果只用于展示测量方法；量化后的任务质量、吞吐和显存收益必须分别评估，
不能由位宽或 QLoRA 训练结果直接推出。
"""

import sys
import time
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8)


def load_quantized_model(model_path: str, quant_type: str):
    """加载不同量化类型的模型. quant_type in {'fp16', 'int8', 'fp4'}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {"device_map": "auto"}
    if quant_type == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif quant_type == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_type == "fp4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"未知 quant_type: {quant_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer


def benchmark(quant_type: str, model_path: str, prompt: str = "Q: What is 2+2?\nA:") -> dict:
    """加载 + 推理 + 测 VRAM + 测延迟."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model, tokenizer = load_quantized_model(model_path, quant_type)
    load_time = time.perf_counter() - t0

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 测延迟
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - t0) * 1000

    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    # 清理
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    return {
        "quant_type": quant_type,
        "load_time_s": round(load_time, 2),
        "inference_ms": round(inference_ms, 1),
        "vram_gb": round(vram_gb, 3),
        "output": generated[:100],
    }


def main():
    if skip_if_mock("an NVIDIA GPU, bitsandbytes, transformers, and local model weights"):
        return
    check_hardware()

    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default`.",
        )

    print("=== FP4 / INT8 / FP16 量化阶梯 (真实 bitsandbytes) ===\n")

    results = []
    for qtype in ["fp16", "int8", "fp4"]:
        print(f"📊 加载 {qtype}...")
        try:
            r = benchmark(qtype, model_path)
            results.append(r)
            print(f"   load: {r['load_time_s']}s | vram: {r['vram_gb']}GB | 32 tokens: {r['inference_ms']}ms")
            print(f"   out: {r['output']}\n")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {str(e)[:120]}\n")

    # 对比
    fp16 = next((r for r in results if r["quant_type"] == "fp16"), None)
    if fp16 and len(results) >= 2:
        print(f"\n=== 对比 FP16 基线 ({fp16['vram_gb']}GB, {fp16['inference_ms']}ms) ===")
        for r in results:
            if r["quant_type"] != "fp16":
                vram_saving = (1 - r["vram_gb"] / fp16["vram_gb"]) * 100
                speed = r["inference_ms"] / fp16["inference_ms"]
                print(
                    f"  {r['quant_type']}: VRAM 节省 {vram_saving:+.0f}%, "
                    f"延迟 {speed:.2f}x ({r['vram_gb']}GB, {r['inference_ms']}ms)"
                )


if __name__ == "__main__":
    main()
