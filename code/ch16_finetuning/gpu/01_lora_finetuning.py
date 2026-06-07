# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.2.2 LoRA 实战代码
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers>=4.40, peft, datasets, accelerate, bitsandbytes
# run: python 01_lora_finetuning.py --mock
# expected_runtime: ~3-10 min on RTX 4090 (7B model) / <5s for mock
# expected_output: LoRA trainable params printed; mock mode prints training config summary
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.2.2
# Interview hooks:
#   1. LoRA 的核心数学原理？为什么低秩分解 W = W₀ + BA 能工作？
#   2. target_modules 应该如何选择？q/v_proj vs 全部投影层的权衡？
#   3. 如何合并 LoRA 权重到原模型（merge_and_unload）？为什么推理时常用？

"""
使用 PEFT 库进行 LoRA 微调 - 完整实战

运行环境：
    pip install torch transformers peft datasets accelerate bitsandbytes
    # GPU 至少 16GB（7B 模型 bf16 + LoRA）；QLoRA 可在 24GB 消费级 GPU 完成
"""

import os
import argparse


# ========== Mock 模式（无 GPU / 无模型权重时使用）==========
MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_lora_pipeline():
    """在无 GPU / 无下载环境下的 mock 实现，演示 LoRA 关键配置与计算"""
    import math

    # 模拟 7B 模型 + LoRA r=16 的参数量
    d, k, r = 4096, 4096, 16
    full_params = d * k                          # 16,777,216（单层 q_proj）
    lora_params = d * r + r * k                  # 65,536 + 65,536 = 131,072
    ratio = lora_params / full_params

    print("[MOCK] 模拟 LoRA 微调（实际未加载模型）")
    print(f"  基座模型参数量:    {full_params/1e6:.2f}M（单层 q_proj）")
    print(f"  LoRA 参数量:       {lora_params/1e6:.4f}M（r={r}, 缩放={r}/{r}=1.0）")
    print(f"  可训练参数比例:    {ratio*100:.3f}%")
    print(f"  全 32 层合计 ≈ 32 × {lora_params/1e6:.2f}M = {32*lora_params/1e6:.1f}M")
    print("  -> 与教程示例 ~33M trainable params 量级一致")
    print()
    print("[MOCK] TrainingArguments 关键超参：")
    print("  num_train_epochs=3, per_device_train_batch_size=4,")
    print("  gradient_accumulation_steps=4 (有效 batch=16)")
    print("  learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler=cosine")
    print("  bf16=True, optim=paged_adamw_32bit, group_by_length=True")
    print()


def real_lora_pipeline():
    """真实 LoRA 微调（需 GPU + 模型权重）"""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # ========== Step 1: 加载模型和分词器 ==========
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 加载模型（bfloat16 节省显存）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # 自动分配 GPU/CPU
        trust_remote_code=True,
    )

    print(f"模型加载完成，原始参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ========== Step 2: 配置 LoRA ==========
    lora_config = LoraConfig(
        r=16,                        # LoRA 秩
        lora_alpha=32,               # 缩放因子 = alpha / r = 2
        target_modules=[             # 应用 LoRA 的模块（不同模型名称不同）
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 期望输出：trainable params: 33,554,432 || all params: 7,000,000,000 || trainable%: 0.479

    # ========== Step 3: 准备训练数据 ==========
    def format_instruction(sample):
        return (
            f"<|im_start|>system\n你是一个专业的客服助手。<|im_end|>\n"
            f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
        )

    train_data = [
        {
            "instruction": "你们的退换货政策是什么？",
            "output": "我们支持7天无理由退货。商品需保持原状，附完整包装。"
                      "退货申请通过后，款项将在3-5个工作日内原路退回。",
        },
        {
            "instruction": "订单多久能到？",
            "output": "一般下单后24小时内发货，国内快递3-5天送达，"
                      "偏远地区可能需要5-7天。您可以在订单详情页查看实时物流信息。",
        },
    ]

    def preprocess(samples):
        # 注意：datasets.map 会传入单条样本字典（batched=False）
        text = format_instruction(samples)
        model_inputs = tokenizer(
            [text],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    dataset = Dataset.from_list(train_data)
    tokenized_dataset = dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)

    # ========== Step 4: 训练配置 ==========
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        group_by_length=True,
        report_to="none",
    )

    # ========== Step 5: 开始训练 ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()

    # ========== Step 6: 保存 LoRA 权重 ==========
    model.save_pretrained("./lora_output/final")
    tokenizer.save_pretrained("./lora_output/final")

    # ========== Step 7: 推理测试 ==========
    def generate_response(model, tokenizer, instruction):
        prompt = (
            f"<|im_start|>system\n你是一个专业的客服助手。<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()

    test_query = "你们的退换货政策是什么？"
    response = generate_response(model, tokenizer, test_query)
    print(f"Q: {test_query}")
    print(f"A: {response}")

    # ========== Step 8: 合并 LoRA 到原模型 ==========
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./merged_model")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU/model)")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_lora_pipeline()
    else:
        real_lora_pipeline()
