# ---
# shared/env.py
# .env loader + API key helper (Wave 14)
# ---
"""
See: tutorial/Ch15.3 (token 管理), Ch18_LLM工程框架实战 §18.1
"""
import os
import sys
from pathlib import Path
from typing import Optional


def _find_dotenv(start: Path) -> Optional[Path]:
    """向上查找 .env 文件 (最多 5 层, 支持 code/.env, repo/.env, ~/work/.env)."""
    for parent in [start, *start.parents[:5]]:
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
        # 一次性提示, 不重复
        if not os.environ.get("_ENV_LOADED_FROM"):
            os.environ["_ENV_LOADED_FROM"] = str(dotenv_path)
            # 不打印, 避免污染 stderr
except ImportError:
    pass  # dotenv 可选


def get_api_key(provider: str = "openai") -> Optional[str]:
    """获取 LLM provider 的 API key.

    Examples:
        get_api_key("openai")      → OPENAI_API_KEY env var
        get_api_key("anthropic")   → ANTHROPIC_API_KEY env var
        get_api_key("deepseek")    → DEEPSEEK_API_KEY env var
        get_api_key("kimi")        → KIMI_API_KEY env var
        get_api_key("siliconflow") → SILICONFLOW_API_KEY env var
    """
    # Wave 14: 增加国内厂商映射
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "kimi": "KIMI_API_KEY",
        "moonshot": "KIMI_API_KEY",  # alias
        "siliconflow": "SILICONFLOW_API_KEY",
    }
    env_var = key_map.get(provider.lower(), f"{provider.upper()}_API_KEY")
    return os.environ.get(env_var)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Simple env var getter."""
    return os.environ.get(key, default)


def get_provider_from_env() -> str:
    """从 LLM_PROVIDER 环境变量获取默认厂商, 找不到则 'mock'."""
    return os.environ.get("LLM_PROVIDER", "mock").strip().lower() or "mock"


if __name__ == "__main__":
    # 简单测试
    print("OPENAI_API_KEY set:", get_api_key("openai") is not None)
    print("DEEPSEEK_API_KEY set:", get_api_key("deepseek") is not None)
    print("KIMI_API_KEY set:", get_api_key("kimi") is not None)
    print("SILICONFLOW_API_KEY set:", get_api_key("siliconflow") is not None)
    print("Working dir:", os.getcwd())
    if dotenv_path := _find_dotenv(Path(__file__).resolve().parent):
        print(f".env loaded from: {dotenv_path}")
