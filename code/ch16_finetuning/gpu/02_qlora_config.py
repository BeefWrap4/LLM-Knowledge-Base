# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.2.3 QLoRA: 4-bit 量化 + LoRA
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers>=4.40, peft, bitsandbytes, accelerate
# run: python 02_qlora_config.py --mock
# expected_runtime: <5s for mock / 3-10 min for real (RTX 3090/4090)
# expected_output: BitsAndBytesConfig + 显存估算 summary
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.2.3
# Interview hooks:
#   1. NF4 量化为什么比 INT4 精度更高？（针对正态分布权重的最优 4-bit 数据类型）
#   2. Double Quantization 如何进一步省显存？Paged Optimizer 防 OOM 的原理？
#   3. QLoRA 的 7B 模型显存构成：4-bit 基础模型 + LoRA + 优化器 + 激活值各占多少？

"""
QLoRA 配置 - 单卡 24GB 消费级 GPU 微调 7B/13B 模型

关键技术：
    1. 4-bit NormalFloat (NF4)：针对正态分布权重设计的最优 4-bit 数据类型
    2. Double Quantization：对量化常数再次量化，再省 ~0.4 bits/param
    3. Paged Optimizer：利用 CPU 内存做 optimizer state offload
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_qlora_config():
    """无 GPU / 无 bitsandbytes 环境的 mock 实现"""
    print("[MOCK] QLoRA 显存估算（7B 模型，bf16 计算 + NF4 存储）")
    print()
    print("  4-bit 基础模型权重:    7B × 0.5B  ≈  3.5 GB")
    print("  LoRA 参数 (r=16):     ~33M × 2B    ≈  0.07 GB  (≈ 70 MB)")
    print("  Adam 优化器状态:       33M × 8B    ≈  0.26 GB  (FP32 m, v)")
    print("  梯度（仅 LoRA）:       33M × 2B    ≈  0.07 GB")
    print("  激活值（512 seq, bs=4）:              ≈  4-8 GB")
    print("  其它（CUDA/框架预留）:                ≈  1-2 GB")
    print("  -------------------------------------------------")
    print("  合计：~ 10-15 GB（单张 RTX 3090/4090 即可）")
    print()
    print("[MOCK] BitsAndBytesConfig 关键参数：")
    print("  load_in_4bit=True")
    print("  bnb_4bit_quant_type='nf4'")
    print("  bnb_4bit_compute_dtype=torch.bfloat16")
    print("  bnb_4bit_use_double_quant=True")
    print()
    print("OK")


def real_qlora_config():
    """真实 QLoRA 配置（需 GPU + bitsandbytes）"""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # 启用 4-bit 量化
        bnb_4bit_quant_type="nf4",               # NF4 量化类型
        bnb_4bit_compute_dtype=torch.bfloat16,   # 计算时用 bf16
        bnb_4bit_use_double_quant=True,          # 二次量化
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 准备模型用于量化训练（必须！）
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置（与 16.2.2 一致）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_qlora_config()
    else:
        real_qlora_config()
