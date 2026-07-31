# ---
# chapter: 19
# topic: Distributed Data Parallel (DDP) 真实训练
# section: 19.2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers, accelerate
# run: accelerate launch --num_processes 1 02_ddp_training.py
#   或 (真实多卡) accelerate launch --num_processes 2 02_ddp_training.py
# expected_runtime: 60-180s (100 steps + model load)
# expected_output: 训练 loss 下降
# ---
# See: ../tutorial/19_分布式训练系统.md §19.2
#
# Interview hooks:
#   1. DDP vs DP (DataParallel)？(答: DDP 多进程无 GIL 竞争, 通信效率高, 单机多卡首选)
#   2. DDP 同步机制？(答: all-reduce 同步梯度, 反向传播结束 barrier 一次)
#   3. 单卡可跑 DDP 吗？(答: 可, accelerate launch --num_processes 1; 真实加速需 2+ GPU)
"""DDP 分布式数据并行训练 (真实 PyTorch DDP).

运行:
  单卡 demo:  accelerate launch --num_processes 1 02_ddp_training.py
  双卡真实:   accelerate launch --num_processes 2 02_ddp_training.py
  4 卡:       accelerate launch --num_processes 4 02_ddp_training.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def check_hardware():
    """DDP 至少 1 卡可跑 (但真实加速需 2+)."""
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


class SyntheticDataset(torch.utils.data.Dataset):
    """合成 dataset (避免下载大语料)."""

    def __init__(self, size: int = 200, seq_len: int = 64):
        self.size = size
        self.seq_len = seq_len
        # 固定 seed 保证所有 rank 看到相同数据
        torch.manual_seed(42)
        # 简单 token IDs (vocab_size = 1000)
        self.data = torch.randint(0, 1000, (size, seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}


def main():
    if skip_if_mock("NVIDIA GPUs, local model weights, and a distributed process group"):
        return
    check_hardware()

    # DDP 初始化 (accelerate 已设 rank/world_size env)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        if local_rank == 0:
            print(f"DDP 初始化: world_size={world_size}, backend=nccl")

    # 加载 Qwen2.5-0.5B (已下载, 1GB)
    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default`.",
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # DDP 包装
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # 合成 dataset + DDP-aware sampler
    dataset = SyntheticDataset(size=200, seq_len=64)
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=(sampler is None))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 训练 100 step
    num_steps = 100
    model.train()
    step = 0
    losses = []
    for epoch in range((num_steps // len(loader)) + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= num_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if step % 10 == 0 and local_rank == 0:
                print(f"[rank {local_rank}] step {step:3d} | loss={loss.item():.4f}")
            step += 1

    # 汇总 (仅 rank 0)
    if local_rank == 0:
        avg_loss = sum(losses) / len(losses)
        print("\n=== 训练完成 ===")
        print(f"  total steps: {len(losses)}")
        print(f"  avg loss: {avg_loss:.4f}")
        print(f"  final loss: {losses[-1]:.4f}")
        if losses[-1] < losses[0]:
            print(f"  loss 下降: {losses[0]:.4f} -> {losses[-1]:.4f}")
        else:
            print("  loss 未下降 (可能需调整 lr / 模型容量)")

    if world_size > 1:
        dist.destroy_process_group()
    print("OK")


if __name__ == "__main__":
    main()
