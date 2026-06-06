# ---
# shared/env.py
# .env loader + API key helper
# ---
"""
See: tutorial/Ch15.3 (token 管理), Ch18_LLM工程框架实战 §18.1
"""
import os
import sys
from pathlib import Path
from typing import Optional


def _find_dotenv(start: Path) -> Optional[Path]:
    """向上查找 .env 文件 (最多 3 层)."""
    for parent in [start, *start.parents[:3]]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


# 模块加载时尝试加载 .env
try:
    from dotenv import load_dotenv

    dotenv_path = _find_dotenv(Path(__file__).resolve().parent)
    if dotenv_path:
        load_dotenv(dotenv_path)
except ImportError:
    pass  # dotenv 可选


def get_api_key(provider: str = "openai") -> Optional[str]:
    """获取 LLM provider 的 API key.

    Examples:
        get_api_key("openai")     → OPENAI_API_KEY env var
        get_api_key("anthropic")  → ANTHROPIC_API_KEY env var
        get_api_key("google")     → GOOGLE_API_KEY env var
    """
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    env_var = key_map.get(provider.lower(), f"{provider.upper()}_API_KEY")
    return os.environ.get(env_var)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Simple env var getter."""
    return os.environ.get(key, default)


if __name__ == "__main__":
    # 简单测试
    print("OPENAI_API_KEY set:", get_api_key("openai") is not None)
    print("Working dir:", os.getcwd())
