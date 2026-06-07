"""Mock implementations for CI/离线测试.

仅 pytest 自动加载 (conftest.py 设 LLM_MOCK=1) 时使用.
主流程不导入此模块 — 如需确定性响应请 from shared._mock_fallback import deterministic_response.
"""
from tests._mocks.mock_llm import MockLLM
from shared._mock_fallback import deterministic_response

__all__ = ["MockLLM", "deterministic_response"]
