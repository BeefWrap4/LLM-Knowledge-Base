# ---
# code/tests/test_gpu_guard.py
# Tests for shared.gpu_guard require_* functions (Wave 1 / Task 2)
# ---
"""
See: tutorial/Ch41_高性能推理引擎与服务 §25.4
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.gpu_guard import (
    require_apple_silicon,
    require_nvidia_gpu,
    require_ollama,
    skip_if_mock,
    skip_unless_apple_silicon,
    skip_unless_enabled,
)


def test_skip_if_mock_from_cli(capsys):
    """显式 --mock 必须在真实硬件调用前返回可识别的 SKIP。"""
    with patch("shared.gpu_guard.sys.argv", ["example.py", "--mock"]):
        assert skip_if_mock("CUDA") is True

    output = capsys.readouterr().out
    assert "[SKIP]" in output
    assert output.rstrip().endswith("OK")


def test_skip_if_mock_disabled():
    """没有 CLI 参数或环境变量时继续真实执行路径。"""
    with (
        patch("shared.gpu_guard.sys.argv", ["example.py"]),
        patch.dict("shared.gpu_guard.os.environ", {}, clear=True),
    ):
        assert skip_if_mock("CUDA") is False


def test_skip_unless_enabled_requires_exact_one(capsys):
    with patch.dict("shared.gpu_guard.os.environ", {"SERVICE_RUN": "true"}, clear=True):
        assert skip_unless_enabled("SERVICE_RUN", "the local service") is True
    output = capsys.readouterr().out
    assert "[SKIP]" in output
    assert "SERVICE_RUN=1" in output
    assert "OK" in output

    with patch.dict("shared.gpu_guard.os.environ", {"SERVICE_RUN": "1"}, clear=True):
        assert skip_unless_enabled("SERVICE_RUN", "the local service") is False


def test_skip_unless_apple_silicon(capsys):
    with (
        patch("shared.gpu_guard._platform_system", return_value="Windows"),
        patch("shared.gpu_guard._platform_machine", return_value="AMD64"),
    ):
        assert skip_unless_apple_silicon() is True
    assert "[SKIP]" in capsys.readouterr().out

    with (
        patch("shared.gpu_guard._platform_system", return_value="Darwin"),
        patch("shared.gpu_guard._platform_machine", return_value="arm64"),
    ):
        assert skip_unless_apple_silicon() is False


def test_require_nvidia_gpu_no_cuda():
    """未检测到 CUDA 时抛 RuntimeError (含中文 + 提示)."""
    with (
        patch("shared.gpu_guard._has_torch", return_value=True),
        patch("shared.gpu_guard._torch_cuda_available", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="需要 NVIDIA GPU"):
            require_nvidia_gpu(min_vram_gb=8)


def test_require_nvidia_gpu_insufficient_vram():
    """显存不足 (4GB < 8GB) 抛 RuntimeError."""
    fake_props = MagicMock(total_memory=4 * 1e9)  # 4GB
    with (
        patch("shared.gpu_guard._has_torch", return_value=True),
        patch("shared.gpu_guard._torch_cuda_available", return_value=True),
        patch("shared.gpu_guard._torch_device_count", return_value=1),
        patch("shared.gpu_guard._torch_device_props", return_value=fake_props),
    ):
        with pytest.raises(RuntimeError, match="显存.*4.0GB.*< 8GB"):
            require_nvidia_gpu(min_vram_gb=8)


def test_require_nvidia_gpu_sufficient():
    """24GB 显存通过检查, 不抛错."""
    fake_props = MagicMock(total_memory=24 * 1e9)
    with (
        patch("shared.gpu_guard._has_torch", return_value=True),
        patch("shared.gpu_guard._torch_cuda_available", return_value=True),
        patch("shared.gpu_guard._torch_device_count", return_value=1),
        patch("shared.gpu_guard._torch_device_props", return_value=fake_props),
    ):
        require_nvidia_gpu(min_vram_gb=8)  # 不抛错


def test_require_apple_silicon_on_linux():
    """非 Darwin 系统 (Linux) 抛 RuntimeError."""
    with patch("shared.gpu_guard._platform_system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="需要 Apple Silicon"):
            require_apple_silicon()


def test_require_ollama_not_running():
    """Ollama 服务不可达抛 RuntimeError."""
    import httpx

    with patch("shared.gpu_guard._httpx_get", side_effect=httpx.ConnectError("ConnectError")):
        with pytest.raises(RuntimeError, match="Ollama 未运行"):
            require_ollama("llama3.2:3b")


def test_require_nvidia_gpu_no_torch():
    """缺 torch → 抛 RuntimeError."""
    with patch("shared.gpu_guard._has_torch", return_value=False):
        with pytest.raises(RuntimeError, match="需要 torch"):
            require_nvidia_gpu(min_vram_gb=8)


def test_require_ollama_missing_model():
    """Ollama 已在跑但缺模型 → 抛错信息包含 '缺模型'."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}
    with patch("shared.gpu_guard._httpx_get", return_value=fake_response):
        with pytest.raises(RuntimeError, match="缺模型 llama3.2:3b"):
            require_ollama("llama3.2:3b")


def test_require_apple_silicon_on_intel_mac():
    """Intel Mac (Darwin + x86_64) → 抛错."""
    with (
        patch("shared.gpu_guard._platform_system", return_value="Darwin"),
        patch("shared.gpu_guard._platform_machine", return_value="x86_64"),
    ):
        with pytest.raises(RuntimeError, match="需要 Apple Silicon"):
            require_apple_silicon()
