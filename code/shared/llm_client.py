# ---
# shared/llm_client.py
# 统一 LLM 客户端 — OpenAI SDK + provider 路由
# 自动注入 base_url + api_key, 缺 Key 时降级到 MockLLM
# ---
"""
统一 LLM 客户端 (drop-in 替代 openai.OpenAI).

设计目标:
  - 用户无需直接 import openai / 配置 base_url
  - 自动从 .env 加载 API Key (经 shared/env.py)
  - 缺 Key 时降级到 MockLLM, 打印 [WARN]
  - 支持 1 行切换厂商: UnifiedClient(provider="kimi")

Usage:
    from shared.llm_client import UnifiedClient
    client = UnifiedClient()                       # 用默认厂商
    resp = client.chat("讲个笑话", system="你是段子手")
    print(resp.content)

    # 显式指定:
    client = UnifiedClient(provider="kimi", model="moonshot-v1-128k")
    resp = client.chat("分析这份 10 万字报告")
"""
import os
import sys
from typing import Any, Optional

from shared.provider_registry import (
    Provider, get_default_provider, get_provider, PROVIDERS,
)
from shared.mock_llm import deterministic_response

# OpenAI SDK 是可选依赖 (anthropic 走自己的 SDK)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class UnifiedClient:
    """统一 LLM 客户端 — 包装 OpenAI SDK, 跨厂商统一接口."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.provider: Provider = get_provider(provider) if provider else get_default_provider()
        self.model: str = model or self.provider.default_chat
        self.api_key: str = api_key or os.environ.get(self.provider.env_key, "")
        self.timeout = timeout

        # LLM_MOCK=1 环境变量 → 强制走 mock (CI/离线)
        if os.environ.get("LLM_MOCK") == "1":
            self.client = None
            return
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            from shared._error_helper import raise_with_help
            raise_with_help(
                f"厂商 {self.provider.name} 缺 API Key (env {self.provider.env_key}).",
                "运行 `make llm-doctor` 诊断; 或参考 README §环境配置.",
            )
        if self.provider.api_style == "openai" and HAS_OPENAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider.base_url,
                timeout=timeout,
            )
        elif self.provider.api_style == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key, timeout=timeout)
            except ImportError:
                from shared._error_helper import raise_with_help
                raise_with_help(
                    f"厂商 {self.provider.name} 需 anthropic SDK.",
                    "运行 `pip install anthropic`.",
                )
        else:
            from shared._error_helper import raise_with_help
            raise_with_help(
                f"厂商 {self.provider.name} 不支持或 openai SDK 缺失.",
                "运行 `make install-llm`.",
            )

    @property
    def is_mock(self) -> bool:
        return self.client is None

    def chat(
        self,
        prompt: str = "",
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        """统一 chat 接口.

        Args:
            prompt: 简单文本输入 (None 时必须给 messages)
            system: system prompt
            messages: 完整 messages 列表 (覆盖 prompt)
            model: 覆盖默认模型
            temperature: 0-2
            max_tokens: 最大输出 token

        Returns:
            对象带 .content (str) / .usage (dict) / .raw (原始响应)
        """
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        if not messages:
            raise ValueError("必须提供 prompt 或 messages")

        used_model = model or self.model

        # Mock 模式
        if self.is_mock:
            content = deterministic_response(messages[-1].get("content", ""), max_length=max_tokens)
            return _LLMResponse(
                content=content,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw=None,
                model=f"mock/{used_model}",
                provider=self.provider.name,
                mock=True,
            )

        # Real call (OpenAI 协议)
        try:
            response = self.client.chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            return _LLMResponse(
                content=content,
                usage=usage,
                raw=response,
                model=used_model,
                provider=self.provider.name,
                mock=False,
            )
        except Exception as e:
            print(f"[ERROR] UnifiedClient: {type(e).__name__}: {e}", file=sys.stderr)
            # 降级到 mock
            return _LLMResponse(
                content=f"[API ERROR, fallback to mock] {deterministic_response(messages[-1].get('content', ''))}",
                usage={},
                raw=e,
                model=f"error/{used_model}",
                provider=self.provider.name,
                mock=True,
            )


class _LLMResponse:
    """统一响应对象 — 简化访问."""
    def __init__(self, content, usage, raw, model, provider, mock):
        self.content = content
        self.usage = usage
        self.raw = raw
        self.model = model
        self.provider = provider
        self.mock = mock

    def __repr__(self):
        return f"<LLMResponse provider={self.provider} model={self.model} mock={self.mock} content[:50]={self.content[:50]!r}>"


# ────────────────────────────────────────────────────────────────
# 便捷函数 — 推荐使用 (1 行)
# ────────────────────────────────────────────────────────────────

def quick_chat(prompt: str, system: Optional[str] = None, **kwargs) -> str:
    """最简调用: quick_chat("讲个笑话") -> str."""
    return UnifiedClient().chat(prompt=prompt, system=system, **kwargs).content


if __name__ == "__main__":
    # 调试: python -m shared.llm_client "你的 prompt"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好, 用一句话介绍你自己"
    client = UnifiedClient()
    print(f"[provider={client.provider.name}, model={client.model}, mock={client.is_mock}]")
    resp = client.chat(prompt=prompt)
    print(f"\n{resp.content}\n")
    if not resp.mock:
        print(f"[usage: {resp.usage}]")
