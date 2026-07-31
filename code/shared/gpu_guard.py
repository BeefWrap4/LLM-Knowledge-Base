# ---
# shared/gpu_guard.py
# CUDA / MPS / CPU 调度器 — 友好报错而非 stack trace
# ---
"""
See: tutorial/Ch11_深度学习与PyTorch, Ch25_推理引擎与高性能服务 §25.4
"""

import os
import shutil
import sys


def skip_if_mock(requirement: str) -> bool:
    """在显式 mock 模式下跳过真实硬件或本地服务调用。

    ``run_all_examples.py --tier gpu`` 会向 GPU 示例传入 ``--mock``。
    示例必须在接触 CUDA、模型文件、浏览器或本地推理服务之前调用本函数，
    从而让离线 CI 清楚地区分“条件性跳过”和“真实运行通过”。

    Args:
        requirement: 面向读者的真实运行前置条件说明。

    Returns:
        ``True`` 表示调用方应立即从 ``main()`` 返回；否则继续真实运行。
    """
    mock_requested = "--mock" in sys.argv or os.environ.get("GPU_MOCK") == "1"
    if not mock_requested:
        return False

    print(f"[SKIP] GPU mock mode: real execution requires {requirement}.")
    print("OK")
    return True


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
        sys.exit("❌  torch not installed. Run: pip install -r requirements-gpu.txt")

    import torch

    if not torch.cuda.is_available():
        sys.exit(
            "❌  torch.cuda.is_available() is False.\n   笔记本/Mac 正常. 跳过 gpu tier, 跑 core/ 或 llm/。"
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


# =============================================================================
# Wave 1 / Task 2: require_* functions for hardware-specific examples
# 供 ch25/ch26/ch28/ch19 等 GPU/Ollama 例子 import 使用.
# 与上面的 require_cuda() (Wave 0 旧接口) 并存, 不破坏现有调用.
# =============================================================================


# --- 私有 helper: 可被 unittest.mock.patch 直接替换 ---


def _has_torch() -> bool:
    """检查 torch 是否已安装."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_cuda_available() -> bool:
    """torch.cuda.is_available() 包装 (用于 mock)."""
    import torch

    return torch.cuda.is_available()


def _torch_device_count() -> int:
    """torch.cuda.device_count() 包装 (用于 mock)."""
    import torch

    return torch.cuda.device_count()


def _torch_device_props(idx: int):
    """torch.cuda.get_device_properties(idx) 包装 (用于 mock)."""
    import torch

    return torch.cuda.get_device_properties(idx)


def _platform_system() -> str:
    """platform.system() 包装 (用于 mock)."""
    import platform

    return platform.system()


def _platform_machine() -> str:
    """platform.machine() 包装 (用于 mock)."""
    import platform

    return platform.machine()


def _httpx_get(url: str, timeout: float = 2.0):
    """httpx.get 包装 (用于 mock + 避免未安装 httpx 时崩溃)."""
    import httpx

    return httpx.get(url, timeout=timeout)


# --- 公共 API: 三种硬件前置条件检查 ---


def require_nvidia_gpu(min_vram_gb: int = 8, min_count: int = 1) -> None:
    """检查 NVIDIA GPU + 显存 + 数量. 不足抛 RuntimeError (带 [HELP] 提示).

    Args:
        min_vram_gb: 每张 GPU 最小显存 (GB). 设为 0 可跳过显存检查.
        min_count: 最少 GPU 张数. 默认 1.

    Raises:
        RuntimeError: 缺 torch / 无 CUDA / 卡数不够 / 显存不足.
    """
    from shared._error_helper import raise_with_help

    if not _has_torch():
        raise_with_help(
            "此例子需要 torch. 运行 `make install-gpu` 安装 GPU 依赖.",
            "无 torch 时无法检测 GPU. 详见 README §硬件 × 章节矩阵.",
        )
    if not _torch_cuda_available():
        raise_with_help(
            f"此例子需要 NVIDIA GPU (≥{min_count} 张, ≥{min_vram_gb}GB). 当前未检测到 CUDA.",
            "详见 README §硬件 × 章节矩阵. Mac/CPU 笔记本可改跑 core/ 或 llm/ 例子.",
        )
    count = _torch_device_count()
    if count < min_count:
        raise_with_help(
            f"需要 ≥{min_count} 张 GPU, 当前 {count} 张.",
            "详见 README §硬件 × 章节矩阵.",
        )
    if min_vram_gb:
        for i in range(count):
            total = _torch_device_props(i).total_memory / 1e9
            if total < min_vram_gb:
                raise_with_help(
                    f"GPU {i} 显存 {total:.1f}GB < {min_vram_gb}GB.",
                    "需要更大显存. 详见 README §硬件矩阵.",
                )


def require_apple_silicon() -> None:
    """检查 Apple Silicon (M-series Mac). 否则抛 RuntimeError.

    Raises:
        RuntimeError: 非 Darwin/arm64 平台.
    """
    from shared._error_helper import raise_with_help

    if _platform_system() != "Darwin" or _platform_machine() != "arm64":
        raise_with_help(
            "此例子需要 Apple Silicon (M-series Mac).",
            "详见 README §硬件 × 章节矩阵. 可改用云端 GPU 或 CPU 例子替代.",
        )


def require_ollama(model: str = "llama3.2:3b") -> None:
    """检查 Ollama 服务 + 指定模型. 否则抛 RuntimeError.

    Args:
        model: 期望已 pull 的模型名 (e.g. "llama3.2:3b").

    Raises:
        RuntimeError: Ollama 未运行 / 模型缺失.
    """
    import httpx

    from shared._error_helper import raise_with_help

    try:
        r = _httpx_get("http://localhost:11434/api/tags", timeout=2.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(model in m for m in models):
            raise_with_help(
                f"Ollama 已运行, 但缺模型 {model}.",
                f"运行 `ollama pull {model}` 拉取模型.",
            )
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        raise_with_help(
            "Ollama 未运行. 先 `ollama serve` 启动服务.",
            "或使用云端 LLM 替代. 详见 README §硬件 × 章节矩阵.",
        )
