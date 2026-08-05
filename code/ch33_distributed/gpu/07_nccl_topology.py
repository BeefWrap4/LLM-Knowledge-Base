# ---
# chapter: 33
# topic: 大模型分布式训练
# topic_id: distributed.nccl_topology
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 07_nccl_topology.py
# expected_runtime: <2s
# expected_output: NCCL version + topology info (or mock)
# ---
# See: ../../../33_大模型分布式训练.md
#
# Interview hooks:
# 1. NCCL 会根据什么条件在 Ring 和 Tree 算法之间切换?
# 2. NVLink, InfiniBand, RoCE 三种互联的典型带宽分别是多少?
# 3. 哪些 NCCL 环境变量对跨节点 InfiniBand 性能影响最大?


try:
    import torch.distributed as dist
except ImportError:
    print("[SKIP] 需要 torch；请安装 GPU tier 依赖")
    print("OK")
    raise SystemExit(0)


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
        import torch as _torch

        if hasattr(_torch, "cuda") and _torch.cuda.is_available():
            print(f"NCCL version: {_torch.cuda.nccl.version()}")
        else:
            print("NCCL version: <unavailable in mock/CPU mode>")
    except Exception as e:
        print(f"NCCL version: <error: {e}>")

    # NCCL 会结合拓扑、collective、消息大小和环境变量选择算法/协议；
    # 不能把 NVSwitch 简化成“必然使用 Tree”。
    print("\nNCCL 内部通信算法选择:")
    print("  - Ring / Tree: 由 NCCL 根据拓扑与消息大小自动选择")
    print("  - CollNet / NVLS 等: 仅在硬件、拓扑和版本支持时可用")


def _print_nccl_summary():
    print("=" * 60)
    print("NCCL 通信拓扑感知 (CPU/Mock 模式):")
    print("=" * 60)
    print("互联规格示例（注意 GB/s 与 Gb/s，不等同于实测 collective 带宽）:")
    print("  - H100 SXM NVLink: 单 GPU 双向聚合标称 900 GB/s")
    print("  - InfiniBand NDR: 常见单端口标称 400 Gb/s（约 50 GB/s）")
    print("  - RoCE: 取决于网卡代际与端口配置，必须查实际集群规格")
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
