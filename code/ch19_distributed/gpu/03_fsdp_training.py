# ---
# chapter: 19
# topic: 分布式训练系统 - FSDP 参数分片训练
# section: 19.2.5
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers (optional for full run)
# run: torchrun --nproc_per_node=8 03_fsdp_training.py
# expected_runtime: 1-3 min (with 8 GPUs)
# expected_output: FSDP-wrapped model + sample loss
# ---
# See: ../tutorial/19_分布式训练系统.md#1924-fsdpfully-sharded-data-parallel
#
# Interview hooks:
# 1. FSDP 相比 DDP 在显存节省上有什么本质差异? 节省的瓶颈在哪?
# 2. FSDP 的 ALL_GATHER 和 REDUCE_SCATTER 分别在什么时机触发?
# 3. auto_wrap_policy 应该如何选择 wrap 的粒度? 太细或太粗会怎样?


# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    print("OK")
    _sys.exit(0)
"""
FSDP 训练示例 - PyTorch 原生参数分片
启动方式: torchrun --nproc_per_node=8 03_fsdp_training.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist


def setup():
    """初始化分布式环境 (mock-friendly)"""
    if "RANK" not in os.environ:
        print("[Mock Mode] Not launched via torchrun.")
        return -1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def train_fsdp():
    local_rank = setup()
    is_mock = local_rank == -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # FSDP 混合精度配置
    # ============================================================
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
            BackwardPrefetch,
            CPUOffload,
        )
    except ImportError:
        print("[Mock Mode] FSDP not available. Showing conceptual demo.")
        _demo_fsdp_concept()
        return

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,   # 参数用 BF16 存储
        reduce_dtype=torch.bfloat16,  # 梯度 reduce 用 BF16
        buffer_dtype=torch.bfloat16,  # buffer 用 BF16
    )

    # ============================================================
    # FSDP 分片策略说明
    # ============================================================
    # FULL_SHARD: ZeRO Stage 3 等价 (参数+梯度+优化器全分片)
    # SHARD_GRAD_OP: ZeRO Stage 2 等价 (仅分片梯度+优化器)
    # HYBRID_SHARD: 节点内全分片, 节点间 DDP
    # NO_SHARD: 退化为 DDP
    sharding_strategy = ShardingStrategy.FULL_SHARD
    print(f"Sharding strategy: {sharding_strategy}")

    # ============================================================
    # 用一个简单模型演示 (生产环境会换成 AutoModelForCausalLM)
    # ============================================================
    class Block(nn.Module):
        """模拟 Transformer Block (作为 FSDP 包装单元)"""
        def __init__(self, dim=512):
            super().__init__()
            self.lin1 = nn.Linear(dim, dim * 4)
            self.lin2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))

    class StackedModel(nn.Module):
        def __init__(self, n_layers=4, dim=512):
            super().__init__()
            self.blocks = nn.ModuleList([Block(dim) for _ in range(n_layers)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    model = StackedModel().to(device)

    if not is_mock:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={Block}
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取下一层参数
            cpu_offload=CPUOffload(offload_params=False),     # 是否卸载到 CPU
            device_id=local_rank,
            # 关键: 限制 FSDP 单元内的参数数量
            limit_all_gathers=True,
        )
    print(f"Model wrapped with FSDP on rank {dist.get_rank() if dist.is_initialized() else 'mock'}")

    # 训练循环 (与 DDP 类似, 但 FSDP 自动管理参数收集和释放)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(10):
        x = torch.randn(2, 512).to(device)
        out = model(x)
        loss = out.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 2 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


def _demo_fsdp_concept():
    """当 FSDP 不可用时的概念演示"""
    print("=" * 60)
    print("FSDP (Fully Sharded Data Parallel) 核心要点:")
    print("=" * 60)
    print("  - 相当于 PyTorch 原生实现的 ZeRO Stage 3")
    print("  - 每个 GPU 只持有 1/N 的参数, 梯度, 优化器状态")
    print("  - 前向/反向时通过 AllGather 动态收集完整参数")
    print("  - 用完后立即释放非本地的参数分片")
    print("  - 关键配置: sharding_strategy / mixed_precision / auto_wrap_policy")
    print("=" * 60)


if __name__ == "__main__":
    train_fsdp()
    print("OK")