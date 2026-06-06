# ---
# shared/gpu_guard.py
# CUDA / MPS / CPU 调度器 — 友好报错而非 stack trace
# ---
"""
See: tutorial/Ch11_深度学习与PyTorch, Ch25_推理引擎与高性能服务 §25.4
"""
import shutil
import sys
from typing import Optional


def require_cuda(min_gb: float = 0) -> dict:
    """检查 CUDA 可用性. 不可用时输出清晰错误并退出.

    Returns:
        dict with 'device' and 'free_gb' keys

    Raises:
        SystemExit: 如果没 CUDA 或显存不足
    """
    if not shutil.which("nvidia-smi"):
        sys.exit(
            "❌  This example needs an NVIDIA GPU + CUDA.\n"
            "   Tip: 在 Mac/笔记本上跳过 gpu tier, 改跑 core/ 或 llm/ 例子。\n"
            "   See QUICKSTART.md §6。"
        )

    try:
        import torch  # noqa: F401  - heavy import lazy
    except ImportError:
        sys.exit(
            "❌  torch not installed. Run: pip install -r requirements-gpu.txt"
        )

    import torch

    if not torch.cuda.is_available():
        sys.exit(
            "❌  torch.cuda.is_available() is False.\n"
            "   笔记本/Mac 正常. 跳过 gpu tier, 跑 core/ 或 llm/。"
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1e9
    total_gb = total_bytes / 1e9
    if free_gb < min_gb:
        sys.exit(
            f"❌  Need ≥{min_gb} GB free VRAM, but only {free_gb:.1f} GB available "
            f"(of {total_gb:.1f} GB total)."
        )

    return {
        "device": torch.cuda.get_device_name(0),
        "free_gb": round(free_gb, 1),
        "total_gb": round(total_gb, 1),
    }


def gpu_summary() -> str:
    """Return a one-line summary of GPU status."""
    if not shutil.which("nvidia-smi"):
        return "GPU: not detected (no nvidia-smi)"
    try:
        import torch
        if not torch.cuda.is_available():
            return "GPU: not available (CPU mode)"
        return (
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"{torch.cuda.device_count()} device(s) | "
            f"compute capability {torch.cuda.get_device_capability(0)}"
        )
    except ImportError:
        return "GPU: torch not installed"


if __name__ == "__main__":
    print(gpu_summary())
