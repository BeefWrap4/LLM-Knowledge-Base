# ---
# chapter: 33
# topic: 大模型分布式训练
# topic_id: distributed.comm_overlap_config
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: 无 (配置示例)
# run: python 09_comm_overlap_config.py
# expected_runtime: <1s
# expected_output: config dict + 解释
# ---
# See: ../../../33_大模型分布式训练.md
#
# Interview hooks:
# 1. overlap_comm 和 contiguous_gradients 在 ZeRO 中分别起到什么作用?
# 2. FSDP 的 BACKWARD_PRE 和 BACKWARD_POST 预取策略区别?
# 3. reduce_bucket_size 设得过大会怎样? 设得过小又会怎样?
# DeepSpeed 通信重叠配置
DEEPSPEED_COMM_OVERLAP_CONFIG = {
    "zero_optimization": {
        "overlap_comm": True,  # ✅ 启用通信重叠
        "reduce_bucket_size": 5e8,  # 示例值；更大不必然更快，也会增加峰值内存/等待时间
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
        forward_prefetch=True,  # 适合静态图且 CPU 调度受限场景，需 profile 决定
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
    print("  forward_prefetch=True  # 仅作静态图配置示例，生产需 profile")
    print("=" * 60)
    print("OK")


if __name__ == "__main__":
    main()
