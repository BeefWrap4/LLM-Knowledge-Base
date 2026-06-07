# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.1 多模态 LoRA 微调 - QLoRA + LoraConfig
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, peft, transformers
# run: python 06_lora_finetune.py
# expected_runtime: <5s (mock) / 5min+ (real)
# expected_output: 演示多模态 LLM 的 LoRA 注入流程
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-1-多模态lora微调
# Interview hooks:
#   1. 多模态微调中为什么视觉编码器一般冻结？
#   2. QLoRA 相比 LoRA 在显存占用上有什么具体收益？
#   3. target_modules 的选择对 LoRA 效果有什么影响？



# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
import os


def build_lora_config():
    """构造多模态 LLM 微调用的 LoraConfig（不加载真实权重）。"""
    from peft import LoraConfig, TaskType

    # 视觉任务通常需要更高秩
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    return lora_config


def main():
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    if use_mock:
        # 不加载真实模型，演示 LoRA 注入与参数统计
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model, TaskType

        # 构造一个 toy 线性模型模拟 LLM
        class ToyLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.o_proj = nn.Linear(64, 64)
                self.gate_proj = nn.Linear(64, 64)
                self.up_proj = nn.Linear(64, 64)
                self.down_proj = nn.Linear(64, 64)

            def forward(self, x):
                return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))

        base = ToyLLM()
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        peft_model = get_peft_model(base, cfg)
        peft_model.print_trainable_parameters()

        # 模拟"投影层全量微调"的逻辑
        class MMProjector(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(1024, 4096)

        mm_projector = MMProjector()
        for param in mm_projector.parameters():
            param.requires_grad = True
        n_train = sum(p.numel() for p in mm_projector.parameters() if p.requires_grad)
        print(f"mm_projector trainable params: {n_train}")
        print("LoRA injection demo OK")
    else:
        # 真实模式
        import torch
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        peft_model = get_peft_model(llm, build_lora_config())
        peft_model.print_trainable_parameters()


if __name__ == "__main__":
    main()
    print("OK")