"""Pytest fixtures for tests/_mocks/ (CI-only mock infrastructure).

本目录的 MockLLM / deterministic_response 仅在 CI 场景下被引用.
提供 mock_env fixture 让测试显式 opt-in 到 mock 模式.

注意: 不在 conftest 顶层设 LLM_MOCK=1 — 那会影响同 suite 中
需要测 "no key → raise" / "real mode" 行为的单元测试.
CI 调用方应通过 env var 或 mock_env fixture 显式启用.
"""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Opt-in fixture: 设 LLM_MOCK=1 + dummy API keys 让测试走 mock.

    Usage:
        def test_foo(mock_env):
            from shared.llm_client import UnifiedClient
            c = UnifiedClient()  # 走 mock, 不抛错
    """
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dummy-test")
    yield
