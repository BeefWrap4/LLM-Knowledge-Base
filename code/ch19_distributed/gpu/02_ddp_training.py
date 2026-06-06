# ---
# chapter: 19
# topic: 分布式训练系统 - DDP 完整训练示例
# section: 19.2.5
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: torchrun --nproc_per_node=8 02_ddp_training.py
# expected_runtime: 1-3 min (with 8 GPUs)
# expected_output: Rank-aware training logs
# ---
# See: ../tutorial/19_分布式训练系统.md#1925-ddp--fsdp-代码示例
#
# Interview hooks:
# 1. DDP 为什么使用多进程而不是多线程? GIL 在这里的影响是什么?
# 2. DDP 的梯度同步发生在 loss.backward() 之前还是之后? 是同步还是异步?
# 3. DistributedSampler 在 DDP 中扮演什么角色? 为什么必须调用 set_epoch()?
"""
DDP 训练完整示例 - 单机多卡 / 多机多卡
启动方式: torchrun --nproc_per_node=8 02_ddp_training.py
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


# ============================================================
# 1. 初始化分布式环境
# ============================================================
def setup_distributed():
    """初始化 NCCL 分布式后端 (mock-friendly)"""
    if not dist.is_available():
        print("[Mock Mode] torch.distributed not available.")
        return -1, 1
    if "RANK" not in os.environ:
        # 在没有 torchrun 启动时, 进入 mock 模式
        print("[Mock Mode] Not launched via torchrun. Mocking single-process.")
        return -1, 1
    dist.init_process_group(
        backend="nccl",          # GPU 通信使用 NCCL
        init_method="env://",    # 从环境变量读取配置 (torchrun 自动设置)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# 2. 模型定义 - 简化的 LLM 模拟
# ============================================================
class SimpleLLM(nn.Module):
    """模拟大模型结构 (使用小尺寸方便在 CPU/Mock 模式下也能跑)"""

    def __init__(self, vocab_size=5000, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, memory=torch.zeros_like(x))  # 简化的 decoder
        return self.lm_head(x)


# ============================================================
# 3. 训练主函数 (DDP 方式)
# ============================================================
def train_ddp():
    local_rank, world_size = setup_distributed()
    is_mock = local_rank == -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_mock:
        global_rank = 0
    else:
        global_rank = dist.get_rank()

    print(f"[Rank {global_rank}/{world_size}] Initializing on device {device}")

    # --- 模型 ---
    model = SimpleLLM().to(device)
    if not is_mock:
        # 关键: 使用 DDP 包装模型
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,   # 生产环境设为 False 提升性能
            gradient_as_bucket_view=True,   # 减少内存拷贝
        )

    # --- 数据集 ---
    n_samples, seq_len, vocab_size = 1000, 128, 5000
    data = torch.randint(0, vocab_size, (n_samples, seq_len))
    labels = torch.randint(0, vocab_size, (n_samples, seq_len))
    dataset = TensorDataset(data, labels)
    # 关键: 使用 DistributedSampler 确保每卡看到不同数据
    if not is_mock:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
        )
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # ============================================================
    # 4. 训练循环
    # ============================================================
    model.train()
    for epoch in range(2):
        if not is_mock and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)  # 每个 epoch 重新 shuffle

        for step, (batch, label) in enumerate(dataloader):
            batch = batch.to(device)
            label = label.to(device)

            # 前向
            outputs = model(batch)
            # 简化 loss: 预测均值与目标均值的距离
            loss = (outputs.mean() - label.float().mean()) ** 2

            # 反向 —— DDP 在这里自动进行梯度 AllReduce
            # backward() 过程中梯度同步与计算重叠
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0 and global_rank == 0:
                if not is_mock:
                    # 注意: 每张卡上的 loss 不同 (数据不同),
                    # 需要 all_reduce 求平均才能得到全局 loss
                    loss_tensor = loss.detach().clone()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    print(f"Epoch {epoch}, Step {step}, "
                          f"Global Loss: {loss_tensor.item():.4f}")
                else:
                    print(f"Epoch {epoch}, Step {step}, "
                          f"Loss (mock): {loss.item():.4f}")

    cleanup()
    if global_rank == 0:
        print("DDP 训练完成 (Mock 模式: 未实际启动多进程)")


if __name__ == "__main__":
    train_ddp()
    print("OK")
