# ---
# chapter: 28
# topic: MLX 统一内存 vs PyTorch MPS 数据复制
# section: 28.3.1 MLX vs CoreML vs PyTorch MPS
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: mlx, torch
# run: python 02_mlx_unified_memory.py
# expected_runtime: <1s (mock mode)
# expected_output: 统一内存 0 拷贝 vs MPS 显式 to('mps') 演示
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.3.1
# Interview hooks:
#   1. 为什么统一内存 (Unified Memory) 对 LLM 推理重要?
#   2. PyTorch MPS 后端为什么需要显式 tensor.to('mps')?
#   3. MLX 在哪些场景下比 MPS 性能更好?
"""演示 MLX 统一内存 vs PyTorch MPS 显式数据复制."""
from __future__ import annotations


def mlx_unified_memory_demo() -> None:
    """MLX: CPU/GPU 共享内存, 无需数据复制."""
    print("--- MLX 统一内存 ---")
    # 真实代码 (需要 Apple Silicon):
    # import mlx.core as mx
    # a = mx.array([1.0, 2.0, 3.0])  # 在 CPU 还是 GPU? 都是! 共享同一段内存
    # b = a * 2                      # MLX 自动调度到合适的设备, 无需迁移
    # print(b)                       # 输出 [2, 4, 6], 零拷贝
    print("MLX 数组在 CPU/GPU 间无需显式复制, 自动调度算子")
    print("  - 创建: a = mx.array([1, 2, 3])  # 统一内存分配")
    print("  - 运算: b = a * 2                # MLX runtime 决定 CPU 还是 GPU")
    print("  - 优势: 避免 7B 模型权重 14GB 的 CPU→GPU 复制开销")


def pytorch_mps_demo() -> None:
    """PyTorch MPS: 需显式 tensor.to('mps') 才能在 GPU 运算."""
    print("--- PyTorch MPS 显式复制 ---")
    print("PyTorch 沿用 CUDA 编程模型:")
    print("  - 创建: a = torch.tensor([1, 2, 3])         # 默认在 CPU")
    print("  - 迁移: a = a.to('mps')                    # 显式 CPU→GPU 复制")
    print("  - 运算: b = a * 2                          # 在 MPS 后端")
    print("  - 取回: b = b.cpu().numpy()                # 必须显式拷回 CPU")
    print("  - 缺点: 14GB 权重的 CPU→GPU 复制耗时 1-3 秒")


def benchmark_summary() -> None:
    """对比: 加载 7B Q4 模型 (4GB) 时的延迟."""
    print("--- 性能对比 (7B Q4 加载) ---")
    print("  MLX (Unified Memory):  ~50ms  (mmap 直接映射)")
    print("  MPS  (PyTorch):        ~1200ms (CPU→GPU 显式复制)")
    print("  CPU  (numpy):          ~800ms  (纯内存加载, 无 GPU 加速)")


def main() -> None:
    mlx_unified_memory_demo()
    print()
    pytorch_mps_demo()
    print()
    benchmark_summary()


if __name__ == "__main__":
    main()
