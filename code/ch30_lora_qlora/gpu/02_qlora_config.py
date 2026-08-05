# ---
# chapter: 30
# topic: SFT、LoRA 与 QLoRA
# topic_id: lora_qlora.qlora_config
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: transformers, peft, bitsandbytes, accelerate, torch
# run: python 02_qlora_config.py
# expected_runtime: 60-180s
# expected_output: 4-bit NF4 量化 + LoRA 训练 loss 下降
# ---
# See: ../../../30_SFT_LoRA与QLoRA.md
#
# Interview hooks:
#   1. NF4 量化为什么比 INT4 精度更高？（针对正态分布权重的最优 4-bit 数据类型）
#   2. Double Quantization 如何进一步省显存？Paged Optimizer 防 OOM 的原理？
#   3. QLoRA 的 7B 模型显存构成：4-bit 基础模型 + LoRA + 优化器 + 激活值各占多少？
"""QLoRA 4-bit 量化 + LoRA 训练 (Qwen2.5-0.5B).

QLoRA = 4-bit NF4 量化 (NormalFloat) base + LoRA adapter
- base 模型: 4-bit 量化存储
- LoRA adapter: 16-bit 训练
- 显存节省: 0.5B 模型 4-bit 量化后 ~0.4GB (vs 1GB fp16)
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


class SyntheticDataset(torch.utils.data.Dataset):
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

    def __getitem__(self, i):
        return self.examples[i]


def main():
    if skip_if_mock("an NVIDIA GPU, local model weights, and a writable training output directory"):
        return
    check_hardware()

    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(f"需要 {model_path}", "运行 `make download-models-default`.")

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== QLoRA 4-bit + LoRA 训练 (Qwen2.5-0.5B) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # 双重量化
    )

    print("加载 4-bit NF4 量化 base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"  4-bit 加载后 VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    # QLoRA 必须: prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    print("\n应用 LoRA adapter (r=16, q/k/v/o_proj)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练
    print("\n训练 (50 steps)...")
    training_args = TrainingArguments(
        output_dir=str(_code_root / "models" / "qlora_adapter"),
        num_train_epochs=1,
        max_steps=50,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=1000,
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

    if losses:
        print("\n=== 训练完成 ===")
        print(f"  initial loss: {losses[0]:.4f}")
        print(f"  final loss:   {losses[-1]:.4f}")
        if losses[-1] < losses[0]:
            print(f"  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")
        else:
            print("  ⚠️  loss 未明显下降 (合成随机数据, 这是正常)")

    vram = torch.cuda.max_memory_allocated() / (1024**3)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  peak allocated VRAM: {vram:.2f}GB / device {total_vram:.2f}GB")
    print("  量化收益必须与同模型、序列、batch、checkpointing 和训练步数的基线对照")
    print("OK")


if __name__ == "__main__":
    main()
