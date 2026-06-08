# ---
# chapter: 19
# topic: 分布式训练系统 - 通信重叠优化配置
# section: 19.8.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: 无 (配置示例)
# run: python 09_comm_overlap_config.py
# expected_runtime: <1s
# expected_output: config dict + 解释
# ---
# See: ../tutorial/19_分布式训练系统.md#1983-通信与计算重叠
#
# Interview hooks:
# 1. overlap_comm 和 contiguous_gradients 在 ZeRO 中分别起到什么作用?
# 2. FSDP 的 BACKWARD_PRE 和 BACKWARD_POST 预取策略区别?
# 3. reduce_bucket_size 设得过大会怎样? 设得过小又会怎样?
# DeepSpeed 通信重叠配置
DEEPSPEED_COMM_OVERLAP_CONFIG = {
    "zero_optimization": {
        "overlap_comm": True,  # ✅ 启用通信重叠
        "reduce_bucket_size": 5e8,  # 500MB bucket (更大 = 更好重叠)
        "allgather_bucket_size": 5e8,
        "contiguous_gradients": True,  # ✅ 梯度连续存储
        "round_robin_gradients": True,  # ✅ 轮询梯度分组 (更好的负载均衡)
    }
}


# FSDP 通信重叠配置 (Python 形式)
def make_fsdp_with_prefetch(model, local_rank):
    """
    用 BACKWARD_PRE 在当前层 backward 时预取下一层参数。
    BACKWARD_PRE: 在当前层 backward 时预取下一层参数
    BACKWARD_POST: 在当前层 backward 后预取 (默认)
    预取让 AllGather (收集参数) 与 backward (计算梯度) 重叠
    """
    try:
        from torch.distributed.fsdp import (
            BackwardPrefetch,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )
    except ImportError:
        print("[Mock Mode] FSDP not available.")
        return None

    return FSDP(
        model,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,  # FSDP2 支持前向预取
        device_id=local_rank,
    )


def main():
    print("=" * 60)
    print("DeepSpeed 通信重叠配置:")
    print("=" * 60)
    for k, v in DEEPSPEED_COMM_OVERLAP_CONFIG["zero_optimization"].items():
        print(f"  {k} = {v}")
    print("=" * 60)
    print("FSDP 通信重叠:")
    print("  backward_prefetch=BackwardPrefetch.BACKWARD_PRE")
    print("  forward_prefetch=True  # FSDP2 新特性")
    print("=" * 60)


if __name__ == "__main__":
    main()
