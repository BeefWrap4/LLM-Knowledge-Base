# ---
# tests/test_vllm_compat.py
# Test vllm_compat dispatch logic (Docker / real vllm / friendly error)
# ---
"""
测试 vllm_compat 调度逻辑 — 不需真 vllm._C, 不需 Docker server.

覆盖:
  - VLLM_BASE_URL env var 切换
  - SamplingParams / EngineArgs / AsyncEngineArgs 字段
  - LLM() factory 在 Docker 模式下返回 _CompatLLM
  - AsyncLLMEngine 在 Docker 模式下 from_engine_args 返回 _CompatAsyncLLMEngine
  - 真 vllm 模式在 vllm._C 缺失时 raise_with_help 友好抛错
"""
import importlib
import os
from unittest.mock import patch

import pytest


# ── env var 切换测试 ─────────────────────────────────────────

def test_no_docker_when_env_unset(monkeypatch):
    """VLLM_BASE_URL 未设 → USE_DOCKER=False."""
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_MODEL_ID", raising=False)
    import shared.vllm_compat
    importlib.reload(shared.vllm_compat)
    assert shared.vllm_compat.USE_DOCKER is False
    assert shared.vllm_compat.VLLM_BASE_URL == ""


def test_use_docker_when_env_set(monkeypatch):
    """VLLM_BASE_URL 设了 → USE_DOCKER=True, 末尾 / 被 strip."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/")
    import shared.vllm_compat
    importlib.reload(shared.vllm_compat)
    assert shared.vllm_compat.USE_DOCKER is True
    assert shared.vllm_compat.VLLM_BASE_URL == "http://localhost:8000"


# ── SamplingParams / EngineArgs 字段测试 ──────────────────────

def test_sampling_params_default():
    """SamplingParams 字段可访问, 默认值正确."""
    from shared.vllm_compat import SamplingParams
    sp = SamplingParams()
    assert sp.temperature == 0.7
    assert sp.max_tokens == 64
    assert sp.top_p == 1.0
    assert sp.top_k == -1
    assert sp.stop == []


def test_sampling_params_custom():
    """SamplingParams 自定义字段."""
    from shared.vllm_compat import SamplingParams
    sp = SamplingParams(temperature=0.5, max_tokens=128, top_p=0.9, stop=["</s>"])
    assert sp.temperature == 0.5
    assert sp.max_tokens == 128
    assert sp.top_p == 0.9
    assert sp.stop == ["</s>"]


def test_engine_args_construction():
    """EngineArgs 字段对齐真 vllm API."""
    from shared.vllm_compat import EngineArgs
    args = EngineArgs(model="test-model", max_num_seqs=16, gpu_memory_utilization=0.8)
    assert args.model == "test-model"
    assert args.max_num_seqs == 16
    assert args.gpu_memory_utilization == 0.8
    assert args.tensor_parallel_size == 1  # default
    assert args.enable_expert_parallel is False  # default


def test_async_engine_args_inherits():
    """AsyncEngineArgs 继承 EngineArgs 字段."""
    from shared.vllm_compat import AsyncEngineArgs
    aae = AsyncEngineArgs(model="test", max_model_len=1024, enforce_eager=False)
    assert aae.model == "test"
    assert aae.max_model_len == 1024
    assert aae.enforce_eager is False


# ── LLM factory 路由测试 ─────────────────────────────────────

def test_llm_factory_routes_to_docker(monkeypatch):
    """VLLM_BASE_URL 设了 → LLM() 走 OpenAI 协议, 返回 _CompatLLM.

    不真连 server: 用 patch._CompatLLM.__init__ 跳过网络 client.
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
    import shared.vllm_compat
    importlib.reload(shared.vllm_compat)

    # 不真实例化 OpenAI client — 直接 mock _CompatLLM
    from shared.vllm_compat import _CompatLLM
    with patch.object(_CompatLLM, "__init__", return_value=None):
        llm = shared.vllm_compat.LLM(model="dummy")
    assert isinstance(llm, _CompatLLM)


def test_async_engine_from_engine_args_routes_to_docker(monkeypatch):
    """VLLM_BASE_URL 设了 → AsyncLLMEngine.from_engine_args 走 Docker."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
    import shared.vllm_compat
    importlib.reload(shared.vllm_compat)

    from shared.vllm_compat import AsyncEngineArgs, _CompatAsyncLLMEngine
    with patch.object(_CompatAsyncLLMEngine, "__init__", return_value=None):
        args = AsyncEngineArgs(model="dummy", max_model_len=512)
        engine = shared.vllm_compat.AsyncLLMEngine.from_engine_args(args)
    assert isinstance(engine, _CompatAsyncLLMEngine)


# ── 真 vllm 模式错误测试 ─────────────────────────────────────

def test_real_vllm_mode_raises_helpful_error(monkeypatch):
    """VLLM_BASE_URL 未设 + vllm 不可用 → raise_with_help (含 Docker 提示)."""
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    import shared.vllm_compat
    importlib.reload(shared.vllm_compat)

    # patch 内置 __import__ 让 import vllm 失败
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError) as exc_info:
            shared.vllm_compat._try_import_real_vllm()
        msg = str(exc_info.value)
        assert "vllm._C" in msg
        assert "VLLM_BASE_URL" in msg  # 提示 Docker escape hatch
        assert "Docker" in msg


# ── Mock config 验证 (Docker 模式打印) ───────────────────────

def test_mock_llm_engine_exposes_config(monkeypatch):
    """Docker 模式 _CompatLLM 提供 llm.llm_engine.vllm_config.X 访问.

    7 个 ch25 文件部分会读 llm.llm_engine.vllm_config.cache_config /
    scheduler_config / parallel_config, 必须兼容.
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
    import shared.vllm_compat
    importlib.reload(shared.vllm_compat)

    from shared.vllm_compat import _CompatLLM
    with patch.object(_CompatLLM, "__init__", return_value=None):
        llm = _CompatLLM(model="dummy", max_num_seqs=4, tensor_parallel_size=2)
    # 手动初始化 (因 __init__ 被 patch)
    llm.model = "dummy"
    from shared.vllm_compat import EngineArgs, _MockLLMEngine
    llm.llm_engine = _MockLLMEngine(EngineArgs(model="dummy", max_num_seqs=4, tensor_parallel_size=2))
    cfg = llm.llm_engine.vllm_config
    assert cfg.cache_config.gpu_memory_utilization == 0.5
    assert cfg.scheduler_config.max_num_seqs == 4
    assert cfg.parallel_config.tensor_parallel_size == 2
    assert cfg.parallel_config.enable_expert_parallel is False
