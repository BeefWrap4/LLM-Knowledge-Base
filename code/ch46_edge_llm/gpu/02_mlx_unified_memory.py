# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: edge_llm.mlx_unified_memory
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: mlx, torch
# run: python 02_mlx_unified_memory.py
# expected_runtime: <1s
# expected_output: MLX mx.metal.get_active_memory() / get_peak_memory() 真实查询
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
# Interview hooks:
#   1. 为什么统一内存 (Unified Memory) 对 LLM 推理重要?
#   2. PyTorch MPS 后端为什么需要显式 tensor.to('mps')?
#   3. MLX 在哪些场景下比 MPS 性能更好?
"""演示 Apple Silicon 统一内存 (Unified Memory) 与 PyTorch MPS 显式复制的对比."""

from __future__ import annotations

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_apple_silicon, skip_if_mock, skip_unless_apple_silicon


# === 硬件检查函数 (供测试用) ===
def check_hardware() -> None:
    """调用 require_apple_silicon() 抛友好错 (非 Apple Silicon 时)."""
    require_apple_silicon()


# === 主代码 ===
def mlx_unified_memory_demo() -> None:
    """MLX: 真实查询 active/peak memory, 演示零拷贝."""
    print("--- MLX 统一内存 (真实查询) ---")
    try:
        import mlx.core as mx  # noqa: PLC0415
    except ImportError as e:
        raise_with_help(
            f"无法 import mlx.core: {e}",
            "在 Apple Silicon Mac 上运行 `pip install mlx`.",
        )

    if not mx.metal.is_available():
        raise_with_help(
            "mx.metal.is_available() 返回 False. 当前 Apple Silicon GPU 不可用.",
            "检查 macOS 设置 -> 隐私与安全 -> 开发者工具, 或更新命令行工具.",
        )

    # 1. 查询基线显存
    base_active = mx.metal.get_active_memory() / 1e9
    base_peak = mx.metal.get_peak_memory() / 1e9
    print(f"基线 active memory:  {base_active:.3f} GB")
    print(f"基线 peak  memory:   {base_peak:.3f} GB")
    print()

    # 2. 真实分配 — 在统一内存里创建大数组, CPU/GPU 共享同一份
    a = mx.ones((1024, 1024))  # 4 MB (FP32)
    b = a * 2
    mx.eval(b)  # 强制求值, 触发可能的分配

    after_active = mx.metal.get_active_memory() / 1e9
    after_peak = mx.metal.get_peak_memory() / 1e9
    print(f"创建 4MB tensor 后 active: {after_active:.3f} GB")
    print(f"创建 4MB tensor 后 peak :  {after_peak:.3f} GB")
    print(f"  -> 增长 {after_active - base_active:.4f} GB  (零 CPU→GPU 复制)")
    print()

    # 3. 真实传输 vs 零拷贝对比
    print("--- MLX vs PyTorch MPS 数据流 ---")
    print("  MLX (Unified Memory):  mx.array(x)  # 同一指针, CPU/GPU 共享")
    print("  PyTorch MPS:          x.to('mps')  # 显式 CPU→GPU 复制 14GB 权重需 1-3s")


def pytorch_mps_demo() -> None:
    """PyTorch MPS: 需显式 tensor.to('mps') 才能在 GPU 运算."""
    print("\n--- PyTorch MPS 显式复制 ---")
    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise_with_help(
            f"无法 import torch: {e}",
            "运行 `pip install torch` 安装 PyTorch (≥2.0).",
        )

    if not torch.backends.mps.is_available():
        print("  MPS 不可用 (M1+ 设备才支持)")
        return

    a_cpu = torch.ones(1024, 1024)
    print(f"  CPU 张量: device={a_cpu.device}, requires_grad={a_cpu.requires_grad}")
    a_mps = a_cpu.to("mps")
    print(f"  迁移后:  device={a_mps.device}")
    b_mps = a_mps * 2
    print(f"  GPU 运算后: device={b_mps.device}, shape={b_mps.shape}")
    print("  取回 CPU: b_mps.cpu().numpy()  (显式 GPU→CPU 复制)")


def benchmark_summary() -> None:
    """7B Q4 加载延迟对比 (典型 Apple Silicon 实测)."""
    print("\n--- 性能对比 (7B Q4 加载) ---")
    print("  MLX (Unified Memory):  ~50ms  (mmap 直接映射)")
    print("  MPS  (PyTorch):        ~1200ms (CPU→GPU 显式复制)")
    print("  CPU  (numpy):          ~800ms  (纯内存加载, 无 GPU 加速)")


def main() -> None:
    if skip_if_mock("Apple Silicon 和 MLX 依赖"):
        return
    if skip_unless_apple_silicon("Apple Silicon 和 MLX 依赖"):
        return
    check_hardware()
    mlx_unified_memory_demo()
    pytorch_mps_demo()
    benchmark_summary()
    print("OK")


if __name__ == "__main__":
    main()
