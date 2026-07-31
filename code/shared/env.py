# ---
# shared/env.py
# .env loader + API key helper (Wave 14)
# ---
"""
See: tutorial/Ch15.3 (token 管理), Ch18_LLM工程框架实战 §18.1
"""

import os
from pathlib import Path

_DOTENV_ATTEMPTED = False


def _find_dotenv(start: Path) -> Path | None:
    """向上查找 .env 文件 (最多 5 层, 支持 code/.env, repo/.env, ~/work/.env)."""
    for parent in [start, *start.parents[:5]]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv_if_real() -> Path | None:
    """仅在显式 ``LLM_MOCK=0`` 时尝试读取 .env，且每进程最多一次。

    这保证默认/mock 验收不会因为导入 ``shared`` 而扫描或读取密钥文件。未设置
    ``LLM_MOCK`` 也属于离线模式，即使进程中已有 Key 也不能返回或使用。
    """
    global _DOTENV_ATTEMPTED

    if os.environ.get("LLM_MOCK") != "0" or _DOTENV_ATTEMPTED:
        loaded = os.environ.get("_ENV_LOADED_FROM")
        return Path(loaded) if loaded else None

    _DOTENV_ATTEMPTED = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    dotenv_path = _find_dotenv(Path(__file__).resolve().parent)
    if not dotenv_path:
        return None
    load_dotenv(dotenv_path)
    os.environ.setdefault("_ENV_LOADED_FROM", str(dotenv_path))
    return dotenv_path


def is_real_llm_mode() -> bool:
    """只有精确 ``LLM_MOCK=0`` 才允许读取 Key 或构造真实客户端。"""
    return os.environ.get("LLM_MOCK") == "0"


def get_api_key(provider: str = "openai") -> str | None:
    """获取 LLM provider 的 API key.

    Examples:
        get_api_key("openai")      → OPENAI_API_KEY env var
        get_api_key("anthropic")   → ANTHROPIC_API_KEY env var
        get_api_key("deepseek")    → DEEPSEEK_API_KEY env var
        get_api_key("kimi")        → KIMI_API_KEY env var
        get_api_key("siliconflow") → SILICONFLOW_API_KEY env var
    """
    if not is_real_llm_mode():
        return None
    load_dotenv_if_real()

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
        "minimax": "MINIMAX_API_KEY",
        "abab": "MINIMAX_API_KEY",  # MiniMax 旧模型名 alias
    }
    env_var = key_map.get(provider.lower(), f"{provider.upper()}_API_KEY")
    return os.environ.get(env_var)


def get_env(key: str, default: str | None = None) -> str | None:
    """Simple env var getter."""
    return os.environ.get(key, default)


def get_provider_from_env() -> str:
    """从 LLM_PROVIDER 环境变量获取默认厂商, 找不到则 'mock'."""
    return os.environ.get("LLM_PROVIDER", "mock").strip().lower() or "mock"


if __name__ == "__main__":
    load_dotenv_if_real()
    # 简单测试
    print("OPENAI_API_KEY set:", get_api_key("openai") is not None)
    print("DEEPSEEK_API_KEY set:", get_api_key("deepseek") is not None)
    print("KIMI_API_KEY set:", get_api_key("kimi") is not None)
    print("SILICONFLOW_API_KEY set:", get_api_key("siliconflow") is not None)
    print("Working dir:", os.getcwd())
    if loaded_from := os.environ.get("_ENV_LOADED_FROM"):
        print(f".env loaded from: {loaded_from}")
