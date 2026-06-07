# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.4 多模态 SFT 微调实战 - LLaVA + LoRA + DeepSpeed ZeRO-2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, datasets, transformers, peft
# run: python 07_sft_training.py
# expected_runtime: <5s (mock)
# expected_output: TrainingArguments / collate_fn 结构演示
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-4-多模态微调实战脚本
# Interview hooks:
#   1. DeepSpeed ZeRO-2 在多模态 SFT 中能节省多少显存？
#   2. collate_fn 在多模态场景下需要处理哪些特殊 token？
#   3. gradient_checkpointing 为什么对 ViT-L + 7B LLM 组合至关重要？



# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
import os


def build_collate_fn_template():
    """返回 collate_fn 模板（不实际加载 processor）。"""
    def collate_fn(batch):
        """将图像和对话整理为模型输入（结构示意）"""
        images = []
        conversations = []
        for item in batch:
            images.append(item["image"])
            conv = [
                {"role": "user", "content": f"<image>\n{item['question']}"},
                {"role": "assistant", "content": item["answer"]},
            ]
            conversations.append(conv)
        return {
            "images": images,
            "conversations": conversations,
        }
    return collate_fn


def build_training_args():
    """构造多模态 SFT 用的 TrainingArguments。"""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir="./llava-lora-checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        # DeepSpeed ZeRO-2
        deepspeed={
            "zero_optimization": {"stage": 2},
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 4,
        },
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


def main():
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    if use_mock:
        # 不加载真实模型/数据，演示脚本结构
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        collate_fn = build_collate_fn_template()
        # 演示一次 collate 调用
        fake_batch = [
            {"image": "img_0.png", "question": "图中有几只猫？", "answer": "两只。"},
            {"image": "img_1.png", "question": "描述这张图。", "answer": "夕阳下的海面。"},
        ]
        out = collate_fn(fake_batch)
        print(f"Batch size: {len(out['images'])}")
        print(f"Sample conv roles: {[c['role'] for c in out['conversations'][0]]}")
        try:
            args = build_training_args()
            print(f"Training output dir: {args.output_dir}")
        except Exception as e:
            print(f"TrainingArguments init skipped: {e}")
        print("SFT training script structure demo OK")
    else:
        from datasets import load_dataset
        from transformers import AutoProcessor, AutoModelForVision2Seq, Trainer
        from peft import LoraConfig, get_peft_model

        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=None, device_map="auto"
        )
        lora_config = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
            lora_dropout=0.05, bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainer = Trainer(
            model=model, args=build_training_args(),
            train_dataset=None, data_collator=build_collate_fn_template(),
        )
        # trainer.train()  # 真实训练时取消注释


if __name__ == "__main__":
    main()
    print("OK")