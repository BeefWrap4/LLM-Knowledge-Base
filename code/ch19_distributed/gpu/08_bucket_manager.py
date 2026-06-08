# ---
# chapter: 19
# topic: 分布式训练系统 - DDP 梯度 Bucket 机制
# section: 19.8.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: 无 (纯 stdlib + 概念演示)
# run: python 08_bucket_manager.py
# expected_runtime: <2s
# expected_output: bucket fill/allreduce sequence
# ---
# See: ../tutorial/19_分布式训练系统.md#1983-通信与计算重叠
#
# Interview hooks:
# 1. DDP 的 bucket_size 默认是多少? 调大或调小分别会怎样?
# 2. 通信与计算重叠的"重叠"具体发生在哪两个执行流上? 它们的依赖关系是什么?
# 3. 为什么 ZeRO-3 不能像 DDP 那样完美实现通信-计算重叠?
# DDP 的梯度 Bucket 机制 (源码简化)
class _MiniBucket:
    """简化的梯度 bucket"""

    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.buffer = []
        self.size = 0

    def add(self, grad):
        self.buffer.append(grad)
        # 假设每个 grad 是一个 float32 元素
        self.size += grad

    def is_full(self):
        return self.size >= self.capacity

    def start_async_allreduce(self):
        # 真实实现: 通过 NCCL 异步启动 all_reduce, 不阻塞后续层
        print(f"  [Async AllReduce] bucket 已满 (size={self.size}), 启动异步 AllReduce")


class BucketManager:
    """
    DDP 将梯度按 bucket_size 分组,
    一个 bucket 填满后立即启动异步 AllReduce,
    同时继续计算其他层的梯度。

    效果:
      Backward(Layer 32) → Bucket 满 → Async AllReduce(Layer 32-29)
      Backward(Layer 28)                                          ← 同时进行
      Backward(Layer 27) → Bucket 满 → Async AllReduce(Layer 27-24)
      ...
    """

    def __init__(self, bucket_size_mb=25):
        self.bucket_size = bucket_size_mb * 1024 * 1024  # 25 MB 默认
        self.buckets = []  # 待填充的 bucket

    def add_gradient(self, param_name: str, grad_size: int):
        """将梯度加入对应 bucket"""
        bucket = self._find_bucket(param_name)
        bucket.add(grad_size)
        if bucket.is_full():
            bucket.start_async_allreduce()
            # 真实实现: 异步启动后清空 bucket 以便复用

    def _find_bucket(self, param_name):
        # 真实实现: 按参数名 hash 决定 bucket 归属
        # 这里简化: 始终使用第一个 bucket
        if not self.buckets:
            self.buckets.append(_MiniBucket(self.bucket_size))
        return self.buckets[0]


def main():
    print("=" * 60)
    print("DDP 梯度 Bucket 机制演示 (简化版)")
    print("=" * 60)
    bm = BucketManager(bucket_size_mb=25)
    # 模拟 32 层 Transformer 的反向传播
    # 每层假设产出 5MB 梯度
    for layer in range(32, 0, -1):
        bm.add_gradient(f"layer_{layer}.weight", grad_size=5 * 1024 * 1024)
    print("=" * 60)
    print("结论: 当一个 bucket 填满后, 异步 AllReduce 不阻塞后续层, 实现通信-计算重叠。")


if __name__ == "__main__":
    main()
