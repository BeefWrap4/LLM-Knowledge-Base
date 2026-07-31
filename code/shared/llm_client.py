# ---
# shared/llm_client.py
# 统一 LLM 客户端 — OpenAI-compatible + Anthropic Messages 路由
# 真实模式缺 Key 或调用失败时 fail closed；未设置或非 0 时才返回离线 mock
# ---
"""
统一 LLM 客户端 (drop-in 替代 openai.OpenAI).

设计目标:
  - 用户无需直接 import openai / 配置 base_url
  - 仅在 LLM_MOCK=0 时按需从 .env 加载 API Key (经 shared/env.py)
  - 缺 Key 或真实 API 失败时抛错，不伪装成成功的 mock
  - 支持 1 行切换厂商: UnifiedClient(provider="kimi")

Usage:
    from shared.llm_client import UnifiedClient
    client = UnifiedClient()                       # 用默认厂商
    resp = client.chat("讲个笑话", system="你是段子手")
    print(resp.content)

    # 显式指定:
    client = UnifiedClient(provider="kimi")
    resp = client.chat("分析这份报告")
"""

import os
import sys
from typing import Any

from shared._mock_fallback import deterministic_response
from shared.provider_registry import (
    Provider,
    get_default_provider,
    get_provider,
)

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
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.provider: Provider = get_provider(provider) if provider else get_default_provider()
        self.model: str = model or self.provider.default_chat
        self.timeout = timeout
        self.api_key = ""

        # fail closed: 只有精确 LLM_MOCK=0 才能创建真实客户端。
        if os.environ.get("LLM_MOCK") != "0":
            self.client = None
            return

        from shared.env import load_dotenv_if_real

        load_dotenv_if_real()
        self.api_key = api_key or os.environ.get(self.provider.env_key, "")
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
        system: str | None = None,
        messages: list[dict] | None = None,
        model: str | None = None,
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
            temperature: 具体范围与兼容性由所选 provider/model 决定
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

        if self.provider.api_style == "anthropic":
            return self._chat_anthropic(
                messages=messages,
                used_model=used_model,
                temperature=temperature,
                max_tokens=max_tokens,
                request_options=dict(kwargs),
            )

        # Real call (OpenAI-compatible protocol)
        try:
            request_options = dict(kwargs)
            if self.provider.name == "openai" and used_model.startswith("gpt-5.6"):
                effort = request_options.setdefault("reasoning_effort", "none")
                request_options.setdefault("max_completion_tokens", max_tokens)
                if effort == "none":
                    request_options.setdefault("temperature", temperature)
            else:
                request_options.setdefault("temperature", temperature)
                request_options.setdefault("max_tokens", max_tokens)
            response = self.client.chat.completions.create(
                model=used_model,
                messages=messages,
                **request_options,
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
            raise

    def _chat_anthropic(
        self,
        *,
        messages: list[dict],
        used_model: str,
        temperature: float,
        max_tokens: int,
        request_options: dict[str, Any],
    ):
        """调用 Anthropic Messages API，并转换为统一响应对象。"""
        system_parts: list[str] = []
        anthropic_messages: list[dict] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if not isinstance(content, str):
                    raise ValueError("Anthropic system content 在本统一接口中必须是字符串")
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                anthropic_messages.append({"role": role, "content": content})
            else:
                raise ValueError(
                    f"Anthropic 统一接口不接受 role={role!r}; 工具调用请直接使用官方 SDK 的 content blocks"
                )

        request_options.setdefault("max_tokens", max_tokens)
        request_options.setdefault("temperature", temperature)
        if system_parts:
            request_options.setdefault("system", "\n\n".join(system_parts))

        try:
            response = self.client.messages.create(
                model=used_model,
                messages=anthropic_messages,
                **request_options,
            )
        except Exception as e:
            print(f"[ERROR] UnifiedClient: {type(e).__name__}: {e}", file=sys.stderr)
            raise

        if isinstance(response.content, str):
            content = response.content
        else:
            blocks = response.content if isinstance(response.content, list) else [response.content]
            content = "".join(
                block.text
                for block in blocks
                if getattr(block, "type", None) == "text" and getattr(block, "text", None)
            )
        usage_obj = getattr(response, "usage", None)
        input_tokens = getattr(usage_obj, "input_tokens", 0) if usage_obj else 0
        output_tokens = getattr(usage_obj, "output_tokens", 0) if usage_obj else 0
        return _LLMResponse(
            content=content,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            raw=response,
            model=used_model,
            provider=self.provider.name,
            mock=False,
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


def quick_chat(prompt: str, system: str | None = None, **kwargs) -> str:
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
