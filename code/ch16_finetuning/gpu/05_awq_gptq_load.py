# ---
# chapter: 16
# topic: AWQ / GPTQ 量化加载 (bitsandbytes NF4 + auto_gptq 可选)
# section: 16.5.2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: transformers, bitsandbytes (auto_gptq 可选)
# run: python 05_awq_gptq_load.py
# expected_runtime: 30-60s
# expected_output: 4-bit 量化模型加载 + 推理
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.5.2
#
# Interview hooks:
#   1. AWQ 与 GPTQ 的核心差异？AWQ 为什么能获得更高精度（激活感知）？
#   2. BitsAndBytes NF4 与 GPTQ/AWQ 的差异？运行时量化 vs 后训练量化？
#   3. fuse_layers=True 的作用？ExLlama / Triton 内核如何加速推理？
"""AWQ / GPTQ 量化加载演示.

量化方案对比:
  - AWQ (Activation-aware Weight Quantization): 保护 1% 显著权重
  - GPTQ (Gradient-based Post-training Quantization): 逐层最小化重建误差
  - bitsandbytes NF4: 4-bit NormalFloat, 通用 4-bit 方案

此处用 bitsandbytes NF4 (最易装) + 提及 AWQ/GPTQ; 如 auto_gptq 已装则额外打印.
"""
import sys
from pathlib import Path
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch
from shared.gpu_guard import require_nvidia_gpu
from shared._error_helper import raise_with_help


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


def main():
    check_hardware()

    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要 {model_path}", "运行 `make download-models-default`."
        )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== AWQ/GPTQ 量化加载 (bitsandbytes NF4 fallback) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("注: AWQ/GPTQ 需预量化权重. 此处演示等价 4-bit 方案 (NF4).")
    print()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("加载 4-bit NF4 量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / (1024**3)
    print(f"  VRAM (4-bit): {vram:.2f}GB")
    print(
        f"  0.5B fp16 baseline: ~1.0GB → 4-bit 节省约 {(1 - vram / 1.0) * 100:.0f}%"
    )

    # 推理测试
    prompt = "Q: What is AI?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n推理输出:\n  {response}")

    print(f"\n=== 量化方案对比 ===")
    print(f"  AWQ:    保护 1% 显著权重, 4-bit, 适合 LLM 推理")
    print(f"  GPTQ:   逐层重建误差最小化, 4-bit, 经典 PTQ")
    print(f"  NF4:    NormalFloat 4-bit, 通用 (本 demo 演示)")
    print(f"  HQQ:    Half-Quadratic Quantization, 无需校准集")
    print(f"  AutoGPTQ: GPTQ 工具链, 需预量化模型")

    # 尝试检查 autogptq
    try:
        import auto_gptq

        print(
            f"\n  本环境: auto_gptq {auto_gptq.__version__} 已装, 可加载 GPTQ 模型"
        )
        print(
            "  使用示例: AutoGPTQForCausalLM.from_quantized('TheBloke/Llama-2-7B-GPTQ', use_triton=True)"
        )
    except ImportError:
        print(f"\n  本环境: auto_gptq 未装, 用 bitsandbytes NF4 等价 4-bit 方案")

    try:
        import awq

        print(f"  本环境: awq {awq.__version__} 已装, 可加载 AWQ 模型")
    except ImportError:
        print(f"  本环境: awq 未装 (NF4 已足够演示 4-bit 推理)")


if __name__ == "__main__":
    main()
