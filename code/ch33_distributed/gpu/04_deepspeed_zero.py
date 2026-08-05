# ---
# chapter: 33
# topic: 大模型分布式训练
# topic_id: distributed.deepspeed_zero
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch, deepspeed, transformers
# run: deepspeed --num_gpus=8 04_deepspeed_zero.py --deeps ds_config.json
# expected_runtime: 1-3 min (with 8 GPUs)
# expected_output: ZeRO-stage loss logs
# ---
# See: ../../../33_大模型分布式训练.md
#
# Interview hooks:
# 1. DeepSpeed 与 HuggingFace Trainer 集成时, 谁负责分布式通信? 谁负责 ZeRO 分片?
# 2. model_engine.backward(loss) 和 loss.backward() 在分布式场景下有什么区别?
# 3. 为什么 ZeRO Stage 3 需要 save_checkpoint 而不是普通的 torch.save?


import sys as _sys

"""
DeepSpeed ZeRO Stage 2/3 完整训练示例
启动方式: deepspeed --num_gpus=8 04_deepspeed_zero.py --deepspeed ds_config.json
"""
import argparse
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock


def train_with_deepspeed_api():
    """使用 DeepSpeed 原生 API 进行训练"""
    # 1. 解析 DeepSpeed 配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = _add_config_arguments(parser)
    args = parser.parse_args()

    try:
        import deepspeed
        import torch
    except ImportError:
        print("[Mock Mode] deepspeed/torch not installed. Showing conceptual demo.")
        _demo_deepspeed_concept()
        return

    from transformers import AutoConfig, AutoModelForCausalLM

    # 2. 构建模型 (生产环境会用 Llama-2-7b, 这里用 mock 尺寸)
    try:
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        # 缩小尺寸以便 mock 模式下也能加载
        config.hidden_size = 128
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 256
        model = AutoModelForCausalLM.from_config(config)
    except Exception as e:
        print(f"[Mock Mode] Cannot load model ({e}); using a tiny nn.Module instead.")
        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 64),
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, 1000),
        )

    # 3. DeepSpeed 初始化 (关键步骤)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # DeepSpeed 会自动:
        # - 包装模型为 DeepSpeedEngine
        # - 创建 ZeRO 优化器 (替代 PyTorch 优化器)
        # - 初始化分布式通信
    )

    # 4. 训练循环
    for step in range(50):
        # 准备数据
        input_ids = torch.randint(0, 1000, (4, 64)).to(model_engine.device)

        # 前向传播
        if hasattr(model_engine, "forward"):
            outputs = model_engine(input_ids, labels=input_ids)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        else:
            loss = model_engine(input_ids).mean()

        # 反向传播 (DeepSpeed 自动处理梯度同步和分片)
        model_engine.backward(loss)

        # 参数更新 (DeepSpeed 自动处理优化器状态分片和 offload)
        model_engine.step()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # 保存 Checkpoint
        if step > 0 and step % 25 == 0:
            model_engine.save_checkpoint(f"./checkpoints/step_{step}")


def _add_config_arguments(parser):
    """兼容性: 不同版本的 deepspeed.add_config_arguments"""
    try:
        import deepspeed

        return deepspeed.add_config_arguments(parser)
    except (ImportError, AttributeError):
        return parser


def _demo_deepspeed_concept():
    print("=" * 60)
    print("DeepSpeed ZeRO 三阶段总结:")
    print("=" * 60)
    print("  Stage 1: 分片优化器状态 → 显存 4P + 12P/N")
    print("  Stage 2: 分片梯度 + 优化器 → 显存 2P + 14P/N")
    print("  Stage 3: 分片参数+梯度+优化器 → 显存 16P/N")
    print()
    print("关键 API:")
    print("  - deepspeed.initialize()  # 替代 torch 的 optimizer")
    print("  - model_engine.backward(loss)  # 自动分片梯度")
    print("  - model_engine.step()  # 自动分片优化器状态")
    print("  - model_engine.save_checkpoint()  # 分片式保存")
    print("=" * 60)


# ============================================================
# 方式二: 使用 HuggingFace Trainer + DeepSpeed (推荐)
# ============================================================
def train_with_hf_trainer():
    """使用 HuggingFace Trainer + DeepSpeed 插件 (mock 演示)"""
    print("=" * 60)
    print("HuggingFace Trainer + DeepSpeed 集成示例:")
    print("=" * 60)

    config_code = """
    from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # 等效 batch_size = 4 * 8 * 8GPU = 256
        learning_rate=3e-4,
        warmup_steps=1000,
        max_steps=10000,
        logging_steps=10,
        save_steps=1000,
        fp16=False,
        bf16=True,              # 优先使用 BF16
        deepspeed="ds_config_stage3.json",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        use_cache=False,  # 训练时必须关闭 KV cache
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    """
    print(config_code)
    print("=" * 60)
    print("启动命令: deepspeed --num_gpus=8 train.py --deepspeed ds_config.json")
    print("=" * 60)


if __name__ == "__main__":
    if not skip_if_mock(
        "NVIDIA GPUs, Hugging Face access, a DeepSpeed process group, and checkpoint storage"
    ):
        train_with_deepspeed_api()
        train_with_hf_trainer()
        print("OK")
