# ---
# chapter: 16
# topic: LoRA Fine-tuning (真实 peft + Trainer)
# section: 16.2.2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: transformers, peft, accelerate, torch
# run: python 01_lora_finetuning.py
# expected_runtime: 60-180s
# expected_output: LoRA 训练 loss 下降 + adapter 保存
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.2.2
#
# Interview hooks:
#   1. LoRA 的核心数学原理？为什么低秩分解 W = W₀ + BA 能工作？
#   2. target_modules 应该如何选择？q/v_proj vs 全部投影层的权衡？
#   3. 如何合并 LoRA 权重到原模型（merge_and_unload）？为什么推理时常用？
"""LoRA 微调真实演示 (Qwen2.5-0.5B + peft + Trainer).

LoRA 冻结原模型权重, 只训练低秩 adapter:
  ΔW = A·B, A ∈ R^{d×r}, B ∈ R^{r×k}, r << min(d,k)
- r=8: 参数量 < 0.1% 原模型
- 显存节省: 训练只需存 optimizer 状态 (adapter 权重)
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


class SyntheticDataset(torch.utils.data.Dataset):
    """合成指令数据, 避免下载大语料."""

    def __init__(self, size: int = 100, seq_len: int = 64):
        self.size = size
        torch.manual_seed(42)
        self.examples = [
            {
                "input_ids": torch.randint(0, 1000, (seq_len,)),
                "labels": torch.randint(0, 1000, (seq_len,)),
            }
            for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    check_hardware()

    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default`.",
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== LoRA Fine-tuning (Qwen2.5-0.5B + peft) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    print("加载 base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    print(f"  原始参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print("\n应用 LoRA adapter (r=8, q_proj+v_proj)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练
    print("\n训练 (50 steps)...")
    training_args = TrainingArguments(
        output_dir=str(_code_root / "models" / "lora_adapter"),
        num_train_epochs=1,
        max_steps=50,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=1000,  # 不存中间
        report_to="none",
        bf16=True,
    )

    dataset = SyntheticDataset(size=100)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]

    # 汇总
    if losses:
        print(f"\n=== 训练完成 ===")
        print(f"  initial loss: {losses[0]:.4f}")
        print(f"  final loss:   {losses[-1]:.4f}")
        if losses[-1] < losses[0]:
            print(f"  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")
        else:
            print(f"  ⚠️  loss 未明显下降 (合成随机数据, 这是正常)")

    # VRAM 统计
    vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  peak VRAM: {vram:.2f}GB / 34GB")

    # 保存 adapter
    save_path = str(_code_root / "models" / "lora_adapter")
    model.save_pretrained(save_path)
    print(f"  adapter saved to: {save_path}")


if __name__ == "__main__":
    main()
