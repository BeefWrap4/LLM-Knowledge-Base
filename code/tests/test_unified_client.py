# ---
# code/tests/test_unified_client.py
# Wave 1 / Task 3: UnifiedClient 缺 Key 必须抛错, 不能再静默降级 mock
# ---
"""
覆盖 UnifiedClient.__init__ 的三种入口:
  - 无 Key / 占位 key: 抛 RuntimeError (缺 API Key)
  - 真实 Key: 走 OpenAI SDK, is_mock=False
  - LLM_MOCK=1: 强制 mock, is_mock=True (不抛错)
"""
import sys
import pytest
from unittest.mock import MagicMock

from shared.llm_client import UnifiedClient


def test_unified_client_no_key_raises(monkeypatch):
    """缺 API Key 必须抛 RuntimeError, 不再降级 mock."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MOCK", raising=False)
    with pytest.raises(RuntimeError, match="缺 API Key"):
        UnifiedClient()


def test_unified_client_dummy_key_raises(monkeypatch):
    """占位 key 'YOUR_API_KEY' 也要抛错."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "YOUR_API_KEY")
    monkeypatch.delenv("LLM_MOCK", raising=False)
    with pytest.raises(RuntimeError, match="缺 API Key"):
        UnifiedClient()


def test_unified_client_with_key_succeeds(monkeypatch):
    """有真实 Key 时不抛错."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real-test-key")
    monkeypatch.delenv("LLM_MOCK", raising=False)
    client = UnifiedClient(provider="deepseek")
    assert client.api_key == "sk-real-test-key"
    assert client.is_mock is False


def test_unified_client_mock_env_var(monkeypatch):
    """LLM_MOCK=1 走 mock, 不抛错."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MOCK", "1")
    client = UnifiedClient(provider="deepseek")
    assert client.is_mock is True


def test_unified_client_anthropic_with_key_succeeds(monkeypatch):
    """anthropic provider + 有 Key + SDK 可用 → 成功初始化."""
    # 清理其他 provider 的 key 防止默认 provider 切换
    for k in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY",
              "SILICONFLOW_API_KEY", "MINIMAX_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.delenv("LLM_MOCK", raising=False)

    # Mock anthropic SDK
    fake_anthropic_module_cls = type("FakeAnthropicModule", (), {})
    fake_anthropic_module = fake_anthropic_module_cls()
    fake_anthropic_class = MagicMock()
    fake_anthropic_module.Anthropic = fake_anthropic_class
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    client = UnifiedClient(provider="anthropic")
    assert client.is_mock is False
    assert client.api_key == "sk-ant-test-key"
    fake_anthropic_class.assert_called_once()


def test_unified_client_anthropic_missing_sdk_raises(monkeypatch):
    """anthropic provider + 无 anthropic SDK → 抛错."""
    for k in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY",
              "SILICONFLOW_API_KEY", "MINIMAX_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.delenv("LLM_MOCK", raising=False)

    # 模拟 anthropic SDK 缺失 — 让 import 失败
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    # 用一个会触发 ImportError 的 builtin
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(RuntimeError, match="需 anthropic SDK"):
        UnifiedClient(provider="anthropic")
