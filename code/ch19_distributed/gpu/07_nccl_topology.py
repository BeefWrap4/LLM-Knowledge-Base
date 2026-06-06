# ---
# chapter: 19
# topic: 分布式训练系统 - NCCL 通信拓扑检测
# section: 19.8.2
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 07_nccl_topology.py
# expected_runtime: <2s
# expected_output: NCCL version + topology info (or mock)
# ---
# See: ../tutorial/19_分布式训练系统.md#1982-nccl-通信库
#
# Interview hooks:
# 1. NCCL 会根据什么条件在 Ring 和 Tree 算法之间切换?
# 2. NVLink, InfiniBand, RoCE 三种互联的典型带宽分别是多少?
# 3. 哪些 NCCL 环境变量对跨节点 InfiniBand 性能影响最大?
import torch.distributed as dist


def inspect_nccl_topology():
    """检查 NCCL 通信拓扑"""
    if not dist.is_available():
        print("[Mock Mode] torch.distributed not available.")
        _print_nccl_summary()
        return

    if dist.is_initialized():
        # 获取当前 rank 和 world_size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank {rank}/{world_size}")
    else:
        print("[Mock Mode] dist not initialized. 单进程演示。")
        rank, world_size = 0, 1

    # NCCL 版本
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            print(f"NCCL version: {torch.cuda.nccl.version()}")
        else:
            print("NCCL version: <unavailable in mock/CPU mode>")
    except Exception as e:
        print(f"NCCL version: <error: {e}>")

    # NCCL 会自动检测最优通信路径:
    # 节点内: NVLink → NVSwitch (全互联)
    # 节点间: InfiniBand → 环形拓扑
    # 当检测到 NVSwitch 时, 使用 Tree 算法而非 Ring 算法
    print("\nNCCL 内部通信算法选择:")
    print("  - Ring: 默认算法, 适用于任意拓扑")
    print("  - Tree: 当有 NVSwitch 时使用, 延迟更低")
    print("  - CollNet: 当检测到特定 InfiniBand 拓扑时使用")


def _print_nccl_summary():
    print("=" * 60)
    print("NCCL 通信拓扑感知 (CPU/Mock 模式):")
    print("=" * 60)
    print("互联带宽速查 (H100/A100 节点):")
    print("  - NVLink (节点内):  900 GB/s on H100")
    print("  - InfiniBand NDR:   400 GB/s (跨节点首选)")
    print("  - RoCE:             200-400 GB/s")
    print("  - 以太网:           最慢, 仅小规模使用")
    print("=" * 60)
    print("关键 NCCL 环境变量:")
    print("  NCCL_IB_DISABLE=0        # 启用 InfiniBand")
    print("  NCCL_SOCKET_IFNAME=eth0  # 指定网卡")
    print("  NCCL_IB_HCA=mlx5_0       # 指定 IB 网卡")
    print("  NCCL_P2P_DISABLE=0       # 启用 GPU P2P 直连")
    print("  NCCL_NET_GDR_LEVEL=5     # GPU Direct RDMA 级别")
    print("=" * 60)


if __name__ == "__main__":
    inspect_nccl_topology()
    print("OK")
