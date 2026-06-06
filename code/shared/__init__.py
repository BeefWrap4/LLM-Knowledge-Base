"""shared utilities for the code companion."""
from shared.env import get_api_key, get_env
from shared.gpu_guard import require_cuda, gpu_summary
from shared.mock_llm import MockLLM, deterministic_response

__all__ = [
    "get_api_key",
    "get_env",
    "require_cuda",
    "gpu_summary",
    "MockLLM",
    "deterministic_response",
]
