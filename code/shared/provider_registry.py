# ---
# shared/provider_registry.py
# 厂商 → (base_url, default_chat_model) 映射表
# 三家国内厂商 (DeepSeek / Kimi / SiliconFlow) 都提供 OpenAI 兼容协议
# ---
"""
LLM Provider Registry — 统一多厂商接入.

Usage:
    from shared.provider_registry import get_provider, list_providers, PROVIDERS

    p = get_provider("deepseek")
    print(p.base_url, p.default_chat)

注册项（截至 2026-07-31）:
  - deepseek / kimi / siliconflow / MiniMax: OpenAI-compatible API
  - openai: OpenAI API
  - anthropic: Anthropic Messages API（使用 anthropic SDK）
  - mock: 仅在显式离线模式下使用，无 API Key

模型名、价格、赠送权益和地区可用性会变化，以各厂商官方模型列表与账户页面为准。
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    """LLM 厂商配置."""

    name: str  # 短代码
    display_name: str  # 中文显示名
    base_url: str  # OpenAI 兼容端点
    default_chat: str  # 默认 chat 模型
    default_reasoner: str | None = None  # 推理模型 (可选)
    api_style: str = "openai"  # openai | anthropic | mock
    env_key: str = ""  # 环境变量名
    region: str = "CN"  # CN | US
    free_tier: str = ""  # 历史字段名；内容是价格/可用性核验提示

    def has_key(self) -> bool:
        """Check if the API key is set in environment."""
        from shared.env import is_real_llm_mode

        if not self.env_key or not is_real_llm_mode():
            return False
        from shared.env import load_dotenv_if_real

        load_dotenv_if_real()
        val = os.environ.get(self.env_key, "").strip()
        return bool(val) and val != "YOUR_API_KEY"


# ────────────────────────────────────────────────────────────────
# 厂商注册表 — 单一来源, 任何新厂商只需在这里加一行
# ────────────────────────────────────────────────────────────────
PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_chat="deepseek-v4-flash",
        default_reasoner="deepseek-v4-pro",
        api_style="openai",
        env_key="DEEPSEEK_API_KEY",
        region="CN",
        free_tier="价格与赠送额度以官方 Models & Pricing 页面为准",
    ),
    "kimi": Provider(
        name="kimi",
        display_name="Kimi (月之暗面)",
        base_url="https://api.moonshot.cn/v1",
        default_chat="kimi-k2.5",
        default_reasoner="kimi-k2.5",
        api_style="openai",
        env_key="KIMI_API_KEY",
        region="CN",
        free_tier="模型、价格与赠送额度以 Moonshot 官方模型列表为准",
    ),
    "siliconflow": Provider(
        name="siliconflow",
        display_name="SiliconFlow (硅基流动)",
        base_url="https://api.siliconflow.cn/v1",
        default_chat="Qwen/Qwen3.6-27B",
        default_reasoner="deepseek-ai/DeepSeek-V4-Pro",
        api_style="openai",
        env_key="SILICONFLOW_API_KEY",
        region="CN",
        free_tier="模型上下线、价格与赠送额度以 SiliconFlow /v1/models 与模型广场为准",
    ),
    "MiniMax": Provider(
        name="MiniMax",
        display_name="MiniMax (Coding Plan)",
        base_url="https://api.minimaxi.com/v1",  # 注意域名是 minimaxi (无 s)
        default_chat="MiniMax-M2.7",
        default_reasoner="MiniMax-M2.7",
        api_style="openai",
        env_key="MINIMAX_API_KEY",
        region="CN",
        free_tier="价格、Token Plan 与可用模型以 MiniMax 官方文档为准",
    ),
    "openai": Provider(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_chat="gpt-5.6",
        api_style="openai",
        env_key="OPENAI_API_KEY",
        region="US",
        free_tier="价格、地区与账户可用性以 OpenAI 官方页面为准",
    ),
    "anthropic": Provider(
        name="anthropic",
        display_name="Anthropic Claude",
        base_url="https://api.anthropic.com",
        default_chat="claude-fable-5",
        api_style="anthropic",  # 用 anthropic SDK, 不是 openai
        env_key="ANTHROPIC_API_KEY",
        region="US",
        free_tier="价格、地区与账户可用性以 Anthropic 官方页面为准",
    ),
    "mock": Provider(
        name="mock",
        display_name="Mock (离线测试)",
        base_url="",
        default_chat="mock-llm",
        api_style="mock",
        env_key="",
        region="-",
        free_tier="无 API 调用, 适合 CI / 离线开发",
    ),
}


def list_providers() -> list[Provider]:
    """返回所有厂商列表 (按 CN 优先排序)."""
    return sorted(
        PROVIDERS.values(),
        key=lambda p: (0 if p.region == "CN" else 1, p.name),
    )


def _resolve_provider_key(name: str) -> str | None:
    """Case-insensitive registry lookup while preserving canonical keys."""
    normalized = name.strip().casefold()
    return next((key for key in PROVIDERS if key.casefold() == normalized), None)


def get_provider(name: str) -> Provider:
    """按名称获取厂商配置（大小写不敏感）；未知名称 fail closed。"""
    key = _resolve_provider_key(name)
    if key is None:
        choices = ", ".join(sorted(PROVIDERS, key=str.casefold))
        raise ValueError(f"未知 LLM provider: {name!r}; 可选: {choices}")
    return PROVIDERS[key]


def get_default_provider() -> Provider:
    """从环境变量推断默认厂商.

    优先级:
      1. LLM_MOCK 不是精确的 "0" → mock (默认/CI 离线短路, 不需 Key)
      2. LLM_PROVIDER 环境变量 (显式选 mock 也允许, 但其他厂商需有 Key)
      3. 第一个有 Key 的国内厂商 (deepseek → kimi → siliconflow)
      4. 第一个有 Key 的海外厂商 (openai → anthropic)
      5. 抛 RuntimeError (不再降级 mock)
    """
    from shared._error_helper import raise_with_help

    # 只有精确 LLM_MOCK=0 才能进入真实 provider 解析。
    if os.environ.get("LLM_MOCK") != "0":
        return PROVIDERS["mock"]

    from shared.env import load_dotenv_if_real

    load_dotenv_if_real()
    env_choice = os.environ.get("LLM_PROVIDER", "").strip()
    env_key = _resolve_provider_key(env_choice) if env_choice else None
    if env_choice and env_key is None:
        choices = ", ".join(sorted(PROVIDERS, key=str.casefold))
        raise ValueError(f"LLM_PROVIDER={env_choice!r} 未注册; 可选: {choices}")
    if env_key is not None:
        p = PROVIDERS[env_key]
        if p.name == "mock":
            return p  # 用户显式选 mock
        if not p.has_key():
            raise_with_help(
                f"LLM_PROVIDER={env_choice} 但缺 API Key (env {p.env_key}).",
                "运行 `make llm-doctor` 诊断; 或 `export LLM_MOCK=1`.",
            )
        return p

    for p in list_providers():
        if p.region == "CN" and p.has_key():
            return p
    for p in list_providers():
        if p.region == "US" and p.has_key():
            return p

    raise_with_help(
        "未配置任何 LLM 厂商: 缺 API Key.",
        "运行 `make llm-doctor` 诊断; 或 `export LLM_MOCK=1` 用 mock (仅 CI).",
    )


def available_providers() -> list[Provider]:
    """返回已配置 API Key 的厂商 (供 llm_doctor.py 报告)."""
    return [p for p in PROVIDERS.values() if p.has_key()]


if __name__ == "__main__":
    # 调试用: python -m shared.provider_registry
    print("All providers:")
    for p in list_providers():
        status = "✓" if p.has_key() else " "
        print(f"  [{status}] {p.name:12s} ({p.region})  {p.display_name}")
    print(f"\nDefault: {get_default_provider().name}")
