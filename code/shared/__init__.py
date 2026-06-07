"""shared utilities for the code companion.

注: MockLLM 已迁移到 tests/_mocks/mock_llm.py (CI-only).
主流程需要 deterministic_response 时请 from shared._mock_fallback import deterministic_response.
"""
from shared.env import get_api_key, get_env
from shared.gpu_guard import require_cuda, gpu_summary

__all__ = [
    "get_api_key",
    "get_env",
    "require_cuda",
    "gpu_summary",
]

print("OK")
