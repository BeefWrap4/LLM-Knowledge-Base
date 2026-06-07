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
import pytest

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
