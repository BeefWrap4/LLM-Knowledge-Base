# ---
# code/tests/test_gpu_guard.py
# Tests for shared.gpu_guard require_* functions (Wave 1 / Task 2)
# ---
"""
See: tutorial/Ch25_推理引擎与高性能服务 §25.4
"""
import pytest
from unittest.mock import patch, MagicMock

from shared.gpu_guard import require_nvidia_gpu, require_apple_silicon, require_ollama


def test_require_nvidia_gpu_no_cuda():
    """未检测到 CUDA 时抛 RuntimeError (含中文 + 提示)."""
    with patch("shared.gpu_guard._has_torch", return_value=True), \
         patch("shared.gpu_guard._torch_cuda_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要 NVIDIA GPU"):
            require_nvidia_gpu(min_vram_gb=8)


def test_require_nvidia_gpu_insufficient_vram():
    """显存不足 (4GB < 8GB) 抛 RuntimeError."""
    fake_props = MagicMock(total_memory=4 * 1e9)  # 4GB
    with patch("shared.gpu_guard._has_torch", return_value=True), \
         patch("shared.gpu_guard._torch_cuda_available", return_value=True), \
         patch("shared.gpu_guard._torch_device_count", return_value=1), \
         patch("shared.gpu_guard._torch_device_props", return_value=fake_props):
        with pytest.raises(RuntimeError, match="显存.*4.0GB.*< 8GB"):
            require_nvidia_gpu(min_vram_gb=8)


def test_require_nvidia_gpu_sufficient():
    """24GB 显存通过检查, 不抛错."""
    fake_props = MagicMock(total_memory=24 * 1e9)
    with patch("shared.gpu_guard._has_torch", return_value=True), \
         patch("shared.gpu_guard._torch_cuda_available", return_value=True), \
         patch("shared.gpu_guard._torch_device_count", return_value=1), \
         patch("shared.gpu_guard._torch_device_props", return_value=fake_props):
        require_nvidia_gpu(min_vram_gb=8)  # 不抛错


def test_require_apple_silicon_on_linux():
    """非 Darwin 系统 (Linux) 抛 RuntimeError."""
    with patch("shared.gpu_guard._platform_system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="需要 Apple Silicon"):
            require_apple_silicon()


def test_require_ollama_not_running():
    """Ollama 服务不可达抛 RuntimeError."""
    with patch("shared.gpu_guard._httpx_get", side_effect=Exception("ConnectError")):
        with pytest.raises(RuntimeError, match="Ollama 未运行"):
            require_ollama("llama3.2:3b")
