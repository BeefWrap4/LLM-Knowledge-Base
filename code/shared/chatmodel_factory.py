# ---
# shared/chatmodel_factory.py
# 统一 ChatModel 工厂 — LangChain / LlamaIndex / 纯 OpenAI SDK 都用同一接口
# 一行切换厂商 (deepseek / kimi / siliconflow / MiniMax / openai / anthropic)
# ---
"""
ChatModel 工厂 — 返回 OpenAI 兼容的 chat client.

为什么这个工厂?
  - DeepSeek / Kimi / SiliconFlow / MiniMax 全部提供 OpenAI 兼容 API
  - LangChain 的 ChatOpenAI + 自定义 base_url 是统一入口
  - LlamaIndex 的 OpenAILike 同样
  - 通过 make_chat_model(provider) 一个调用返回正确配置

Usage:
    from shared.chatmodel_factory import make_chat_model, has_langchain, has_llama_index

    # LangChain
    if has_langchain():
        llm = make_chat_model(provider="deepseek", framework="langchain")
        from langchain_core.prompts import ChatPromptTemplate
        chain = ChatPromptTemplate.from_template("讲个笑话") | llm
        print(chain.invoke({}).content)

    # LlamaIndex
    if has_llama_index():
        llm = make_chat_model(provider="kimi", framework="llama_index")
        # ... 用 llm 完成 query engine

    # 纯 OpenAI 兼容
    from shared.chatmodel_factory import make_openai_client
    client = make_openai_client(provider="siliconflow")
    resp = client.chat.completions.create(model="Qwen/Qwen2.5-72B", messages=[...])
"""
from typing import Any, Optional

from shared.provider_registry import (
    PROVIDERS, Provider, get_default_provider, get_provider,
)


# ────────────────────────────────────────────────────────────────
# 可选框架检测
# ────────────────────────────────────────────────────────────────

def has_langchain() -> bool:
    try:
        from langchain_openai import ChatOpenAI  # noqa
        return True
    except ImportError:
        return False


def has_llama_index() -> bool:
    try:
        from llama_index.core.llms import OpenAILike  # noqa
        return True
    except ImportError:
        return False


# ────────────────────────────────────────────────────────────────
# 纯 OpenAI 客户端
# ────────────────────────────────────────────────────────────────

def make_openai_client(provider: Optional[str] = None, **overrides):
    """创建 OpenAI 兼容 SDK client.

    Usage:
        client = make_openai_client()                  # 默认厂商
        client = make_openai_client(provider="kimi")   # 指定厂商
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai (or `make install-llm`)")

    p = get_provider(provider) if provider else get_default_provider()
    api_key = overrides.pop("api_key", None) or _get_key(p)
    if not api_key:
        raise ValueError(
            f"厂商 {p.name} 缺 API Key (env {p.env_key}), "
            f"或显式传 api_key="
        )
    return OpenAI(
        api_key=api_key,
        base_url=overrides.pop("base_url", p.base_url),
        **overrides,
    )


def _get_key(p: Provider) -> Optional[str]:
    import os
    return os.environ.get(p.env_key, "").strip() or None


# ────────────────────────────────────────────────────────────────
# 统一 ChatModel 工厂 (LangChain / LlamaIndex)
# ────────────────────────────────────────────────────────────────

def make_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    framework: str = "langchain",  # "langchain" | "llama_index"
    temperature: float = 0.7,
    **kwargs,
):
    """创建 ChatModel 实例, 跨厂商统一接口.

    Args:
        provider: 厂商名 (deepseek / kimi / siliconflow / MiniMax / openai / anthropic)
        model: 模型名 (None = 用厂商默认)
        framework: "langchain" 或 "llama_index"
        temperature: 0-2
        **kwargs: 传给底层 ChatModel 的额外参数

    Returns:
        LangChain BaseChatModel 或 LlamaIndex LLM 实例.
        若厂商无 API Key, 返回 None (调用方需 SKIP).

    Examples:
        llm = make_chat_model(provider="deepseek")
        llm = make_chat_model(provider="kimi", model="moonshot-v1-128k", framework="llama_index")
    """
    p = get_provider(provider) if provider else get_default_provider()
    api_key = _get_key(p)
    if not api_key:
        # 缺 key, 返回 None — 调用方需 [SKIP]
        return None

    used_model = model or p.default_chat

    if framework == "langchain":
        return _make_langchain(p, used_model, api_key, temperature, **kwargs)
    elif framework == "llama_index":
        return _make_llama_index(p, used_model, api_key, temperature, **kwargs)
    else:
        raise ValueError(f"未知 framework: {framework} (仅支持 langchain / llama_index)")


def _make_langchain(p: Provider, model: str, key: str, temperature: float, **kwargs):
    """创建 LangChain ChatModel.

    OpenAI 兼容厂商 (deepseek/kimi/siliconflow/MiniMax/openai) 全部用 ChatOpenAI + base_url.
    Anthropic 用 ChatAnthropic (独立 SDK).
    """
    if p.api_style == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("pip install langchain-anthropic")
        return ChatAnthropic(
            model=model,
            api_key=key,
            temperature=temperature,
            **kwargs,
        )
    else:
        # OpenAI 兼容: deepseek, kimi, siliconflow, MiniMax, openai
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("pip install langchain-openai")
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=p.base_url,
            temperature=temperature,
            **kwargs,
        )


def _make_llama_index(p: Provider, model: str, key: str, temperature: float, **kwargs):
    """创建 LlamaIndex LLM.

    OpenAI 兼容厂商全部用 OpenAILike + base_url.
    """
    if p.api_style == "anthropic":
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError:
            raise ImportError("pip install llama-index-llms-anthropic")
        return Anthropic(model=model, api_key=key, **kwargs)
    else:
        try:
            from llama_index.llms.openai_like import OpenAILike
        except ImportError:
            try:
                # 旧版本 llama_index 在 core.llms
                from llama_index.core.llms import OpenAILike
            except ImportError:
                raise ImportError("pip install llama-index-llms-openai-like")
        # OpenAILike 是 OpenAI 兼容, 设置 context_window 等
        return OpenAILike(
            model=model,
            api_key=key,
            api_base=p.base_url,
            temperature=temperature,
            is_chat_model=True,
            context_window=kwargs.pop("context_window", 128000),
            **kwargs,
        )


# ────────────────────────────────────────────────────────────────
# 便捷函数
# ────────────────────────────────────────────────────────────────

def quick_chat(prompt: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
    """最简调用: 1 行发请求, 返回 string. 适合教程例子."""
    from shared.llm_client import UnifiedClient
    return UnifiedClient(provider=provider, model=model).chat(prompt=prompt, **kwargs).content


def doctor_summary() -> dict:
    """返回所有可用的厂商 + 框架 + key 状态, 供诊断."""
    return {
        "providers": [
            {
                "name": p.name,
                "display": p.display_name,
                "region": p.region,
                "default_chat": p.default_chat,
                "has_key": p.has_key(),
                "free_tier": p.free_tier,
            }
            for p in PROVIDERS.values()
        ],
        "frameworks": {
            "langchain": has_langchain(),
            "llama_index": has_llama_index(),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(doctor_summary(), indent=2, ensure_ascii=False))
