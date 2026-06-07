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

支持厂商:
  - deepseek      DeepSeek V3 / R1, 强推理, 国内访问快
  - kimi          Moonshot 长上下文 128K, ¥15 体验金
  - siliconflow   多模型路由 (Qwen / GLM / DeepSeek), 性价比高
  - openai        GPT-4o / GPT-4o-mini (海外)
  - anthropic     Claude Sonnet 4 / Opus 4 (海外, 用 anthropic SDK)
  - mock          离线 mock, 无 API Key
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Provider:
    """LLM 厂商配置."""
    name: str                                # 短代码
    display_name: str                        # 中文显示名
    base_url: str                            # OpenAI 兼容端点
    default_chat: str                        # 默认 chat 模型
    default_reasoner: Optional[str] = None   # 推理模型 (可选)
    api_style: str = "openai"                # openai | anthropic | mock
    env_key: str = ""                        # 环境变量名
    region: str = "CN"                       # CN | US
    free_tier: str = ""                      # 免费额度说明

    def has_key(self) -> bool:
        """Check if the API key is set in environment."""
        if not self.env_key:
            return False
        val = os.environ.get(self.env_key, "").strip()
        return bool(val) and val != "YOUR_API_KEY"


# ────────────────────────────────────────────────────────────────
# 厂商注册表 — 单一来源, 任何新厂商只需在这里加一行
# ────────────────────────────────────────────────────────────────
PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_chat="deepseek-chat",                  # V3
        default_reasoner="deepseek-reasoner",        # R1 推理
        api_style="openai",
        env_key="DEEPSEEK_API_KEY",
        region="CN",
        free_tier="注册送 ¥10, ¥1/百万 token",
    ),
    "kimi": Provider(
        name="kimi",
        display_name="Kimi (月之暗面)",
        base_url="https://api.moonshot.cn/v1",
        default_chat="moonshot-v1-8k",
        default_reasoner=None,
        api_style="openai",
        env_key="KIMI_API_KEY",
        region="CN",
        free_tier="新用户 ¥15 体验金, 128K 上下文",
    ),
    "siliconflow": Provider(
        name="siliconflow",
        display_name="SiliconFlow (硅基流动)",
        base_url="https://api.siliconflow.cn/v1",
        default_chat="Qwen/Qwen2.5-7B-Instruct",
        default_reasoner="Qwen/QwQ-32B-Preview",
        api_style="openai",
        env_key="SILICONFLOW_API_KEY",
        region="CN",
        free_tier="注册送 2000 万 tokens, 多模型路由",
    ),
    "MiniMax": Provider(
        name="MiniMax",
        display_name="MiniMax (MiniMax, Codin Plan)",
        base_url="https://api.minimaxi.com/v1",          # 注意域名是 minimaxi (无 s)
        default_chat="MiniMax-Text-01",
        default_reasoner="MiniMax-Text-01",  # same model supports thinking via reasoning_effort param
        api_style="openai",
        env_key="MINIMAX_API_KEY",
        region="CN",
        free_tier="Codin Plan (key prefix sk-cp-): 编码优化订阅, ¥X/年起",
    ),
    "openai": Provider(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_chat="gpt-4o-mini",
        api_style="openai",
        env_key="OPENAI_API_KEY",
        region="US",
        free_tier="需付费 / 信用卡",
    ),
    "anthropic": Provider(
        name="anthropic",
        display_name="Anthropic Claude",
        base_url="https://api.anthropic.com",
        default_chat="claude-sonnet-4-5",
        api_style="anthropic",  # 用 anthropic SDK, 不是 openai
        env_key="ANTHROPIC_API_KEY",
        region="US",
        free_tier="需付费",
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


def get_provider(name: str) -> Provider:
    """按名称获取厂商配置; 找不到时返回 mock."""
    return PROVIDERS.get(name) or PROVIDERS["mock"]


def get_default_provider() -> Provider:
    """从环境变量推断默认厂商.

    优先级:
      1. LLM_MOCK=1 → mock (CI/离线 短路, 不需 Key)
      2. LLM_PROVIDER 环境变量 (显式选 mock 也允许, 但其他厂商需有 Key)
      3. 第一个有 Key 的国内厂商 (deepseek → kimi → siliconflow)
      4. 第一个有 Key 的海外厂商 (openai → anthropic)
      5. 抛 RuntimeError (不再降级 mock)
    """
    from shared._error_helper import raise_with_help

    # LLM_MOCK=1 是 CI 短路标志: 任何缺 Key 场景下返回 mock 而不抛错
    if os.environ.get("LLM_MOCK") == "1":
        return PROVIDERS["mock"]

    env_choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_choice and env_choice in PROVIDERS:
        p = PROVIDERS[env_choice]
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
