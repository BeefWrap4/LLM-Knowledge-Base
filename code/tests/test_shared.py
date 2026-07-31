# ---
# code/tests/test_shared.py (Wave 23)
# 单元测试: shared/ 模块的核心功能
# 用 pytest 跑: pytest tests/test_shared.py -v
# ---
"""测试 provider_registry, llm_client, chatmodel_factory 的核心 API.

不依赖网络/API Key, 验证:
  - PROVIDERS 字典完整性
  - get_provider / get_default_provider
  - UnifiedClient 在无 Key 时降级 mock
  - make_chat_model 在无 Key 时返回 None
  - make_openai_client 在无 SDK 时抛 ImportError
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))


# ═══════════════════════════════════════════════════════════
# provider_registry
# ═══════════════════════════════════════════════════════════


def test_providers_has_seven_entries():
    """PROVIDERS 字典应有 7 个厂商."""
    from shared.provider_registry import PROVIDERS

    assert len(PROVIDERS) == 7, f"期望 7 个厂商, 实际 {len(PROVIDERS)}"


def test_providers_includes_chinese_vendors():
    """应包含 4 个国内厂商 (CN region)."""
    from shared.provider_registry import PROVIDERS

    cn = [p for p in PROVIDERS.values() if p.region == "CN"]
    names = {p.name for p in cn}
    assert "deepseek" in names
    assert "kimi" in names
    assert "siliconflow" in names
    assert "MiniMax" in names


def test_all_providers_have_required_fields():
    """每个 Provider 必须有 name / base_url / env_key / api_style."""
    from shared.provider_registry import PROVIDERS, Provider

    for name, p in PROVIDERS.items():
        assert isinstance(p, Provider), f"{name} 不是 Provider 实例"
        assert p.name, f"{name} name 为空"
        assert p.env_key or p.api_style == "mock", f"{name} env_key 缺失"
        assert p.api_style in ("openai", "anthropic", "mock"), f"{name} api_style 异常"


def test_get_provider_unknown_raises():
    """未知厂商名必须 fail closed，不能静默切到 mock。"""
    from shared.provider_registry import get_provider

    with pytest.raises(ValueError, match="未知 LLM provider"):
        get_provider("nonexistent_vendor_xyz")


def test_get_provider_known():
    """已知厂商名返回正确 Provider."""
    from shared.provider_registry import get_provider

    assert get_provider("deepseek").name == "deepseek"
    assert get_provider("MiniMax").name == "MiniMax"
    assert get_provider("minimax").name == "MiniMax"
    assert get_provider("MINIMAX").name == "MiniMax"
    assert get_provider("kimi").display_name == "Kimi (月之暗面)"


def test_get_default_provider_no_key_returns_mock():
    """LLM_MOCK 未设时默认离线，即使无 Key 也返回 mock。"""
    from shared.provider_registry import get_default_provider

    with patch.dict(os.environ, {}, clear=True):
        assert get_default_provider().name == "mock"


def test_get_default_provider_no_key_with_llm_mock_returns_mock():
    """无 API Key + LLM_MOCK=1 → 返回 mock (CI 短路)."""
    from shared.provider_registry import get_default_provider

    with patch.dict(os.environ, {"LLM_MOCK": "1"}, clear=True):
        assert get_default_provider().name == "mock"


def test_get_default_provider_with_deepseek_key():
    """设 DEEPSEEK_API_KEY + LLM_PROVIDER=deepseek 时默认厂商是 deepseek."""
    from shared.provider_registry import get_default_provider

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "sk-test-xxx",
            "LLM_PROVIDER": "deepseek",
            "LLM_MOCK": "0",
        },
        clear=False,
    ):
        assert get_default_provider().name == "deepseek"


def test_get_default_provider_respects_env_override():
    """LLM_PROVIDER 环境变量优先级最高."""
    from shared.provider_registry import get_default_provider

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "sk-1",
            "KIMI_API_KEY": "sk-2",
            "LLM_PROVIDER": "kimi",
            "LLM_MOCK": "0",
        },
        clear=False,
    ):
        assert get_default_provider().name == "kimi"


def test_get_default_provider_minimax_is_case_insensitive():
    """LLM_PROVIDER=minimax/MiniMax 均应命中规范的 MiniMax provider。"""
    from shared.provider_registry import get_default_provider

    with patch.dict(
        os.environ,
        {
            "MINIMAX_API_KEY": "sk-test",
            "LLM_PROVIDER": "minimax",
            "LLM_MOCK": "0",
        },
        clear=True,
    ):
        assert get_default_provider().name == "MiniMax"


def test_get_default_provider_unknown_env_override_raises():
    """LLM_PROVIDER 拼写错误不能被忽略后切到其他有 Key 的厂商。"""
    from shared.provider_registry import get_default_provider

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "sk-test",
            "LLM_PROVIDER": "deepseak",
            "LLM_MOCK": "0",
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="未注册"):
            get_default_provider()


# ═══════════════════════════════════════════════════════════
# llm_client
# ═══════════════════════════════════════════════════════════


def test_unified_client_unset_mode_defaults_to_mock():
    """未设置 LLM_MOCK 时必须离线，不能因进程 Key 状态而联网。"""
    from shared.llm_client import UnifiedClient

    with patch.dict(os.environ, {}, clear=True):
        assert UnifiedClient(provider="deepseek").is_mock is True


def test_unified_client_no_key_with_mock_env_works():
    """无 API Key + LLM_MOCK=1 → 走 mock, 不抛错."""
    from shared.llm_client import UnifiedClient

    with patch.dict(os.environ, {"LLM_MOCK": "1"}, clear=True):
        c = UnifiedClient(provider="deepseek")
        assert c.is_mock is True
        resp = c.chat(prompt="test")
        assert resp.mock is True
        assert "确定性" in resp.content or "mock" in resp.content.lower() or len(resp.content) > 0


def test_unified_client_with_key_not_mock():
    """有 API Key 时 UnifiedClient 不降级."""
    from shared.llm_client import UnifiedClient

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "sk-test-xxx",
            "LLM_MOCK": "0",
        },
        clear=False,
    ):
        c = UnifiedClient(provider="deepseek")
        assert c.is_mock is False
        assert c.api_key == "sk-test-xxx"
        assert c.client is not None


def test_unified_client_response_has_attrs():
    """_LLMResponse 应有 content / model / provider / mock 字段 (LLM_MOCK=1)."""
    from shared.llm_client import UnifiedClient

    with patch.dict(os.environ, {"LLM_MOCK": "1"}, clear=True):
        c = UnifiedClient()
        resp = c.chat(prompt="hello")
        assert hasattr(resp, "content")
        assert hasattr(resp, "model")
        assert hasattr(resp, "provider")
        assert hasattr(resp, "mock")
        assert hasattr(resp, "usage")


def test_unified_client_chat_with_messages():
    """messages 形式调用应工作 (LLM_MOCK=1)."""
    from shared.llm_client import UnifiedClient

    with patch.dict(os.environ, {"LLM_MOCK": "1"}, clear=True):
        c = UnifiedClient()
        resp = c.chat(
            system="你是一个助手",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        assert resp.mock is True
        assert len(resp.content) > 0


# ═══════════════════════════════════════════════════════════
# chatmodel_factory
# ═══════════════════════════════════════════════════════════


def test_has_langchain():
    """has_langchain 检查 langchain_openai 可导入."""
    from shared.chatmodel_factory import has_langchain

    # 当前环境: langchain_openai 已装 (跑得起其他测试就是装了)
    result = has_langchain()
    assert isinstance(result, bool)


def test_has_llama_index():
    """has_llama_index 检查 llama_index.core 可导入."""
    from shared.chatmodel_factory import has_llama_index

    result = has_llama_index()
    assert isinstance(result, bool)


def test_make_chat_model_no_key_returns_none():
    """无 Key 时返回 None (不抛异常)."""
    from shared.chatmodel_factory import make_chat_model

    with patch.dict(os.environ, {}, clear=True):
        result = make_chat_model(provider="deepseek")
        assert result is None


def test_make_chat_model_mock_blocks_explicit_provider_and_key():
    """LLM_MOCK=1 必须压过显式 provider 和已有 Key，避免离线验收误联网。"""
    from shared.chatmodel_factory import make_chat_model

    with patch.dict(
        os.environ,
        {"LLM_MOCK": "1", "DEEPSEEK_API_KEY": "sk-test-should-not-be-used"},
        clear=False,
    ):
        assert make_chat_model(provider="deepseek") is None


def test_make_openai_client_mock_blocks_explicit_provider_and_key():
    """纯 SDK 工厂也必须尊重 LLM_MOCK=1。"""
    from shared.chatmodel_factory import make_openai_client

    with patch.dict(
        os.environ,
        {"LLM_MOCK": "1", "DEEPSEEK_API_KEY": "sk-test-should-not-be-used"},
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="只有显式 LLM_MOCK=0"):
            make_openai_client(provider="deepseek")


def test_make_openai_client_rejects_anthropic(monkeypatch):
    """Anthropic Messages API 不能误走 OpenAI SDK。"""
    from shared.chatmodel_factory import make_openai_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM_MOCK", "0")
    with pytest.raises(ValueError, match="不是 OpenAI-compatible"):
        make_openai_client(provider="anthropic")


def test_make_chat_model_unknown_framework_raises():
    """未知 framework 抛 ValueError."""
    from shared.chatmodel_factory import make_chat_model

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
        with pytest.raises(ValueError, match="未知 framework"):
            make_chat_model(provider="deepseek", framework="unknown_fw")


def test_make_chat_model_default_provider():
    """不指定 provider + 显式 LLM_PROVIDER=deepseek 时用 deepseek."""
    from shared.chatmodel_factory import make_chat_model

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "sk-test",
            "LLM_PROVIDER": "deepseek",
            "LLM_MOCK": "0",
        },
        clear=False,
    ):
        llm = make_chat_model()
        # langchain ChatOpenAI 实例
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)
        assert "deepseek" in llm.openai_api_base


def test_make_chat_model_minimax():
    """MiniMax provider."""
    from langchain_openai import ChatOpenAI

    from shared.chatmodel_factory import make_chat_model

    with patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "sk-cp-test", "LLM_MOCK": "0"},
        clear=False,
    ):
        llm = make_chat_model(provider="MiniMax")
        assert isinstance(llm, ChatOpenAI)
        assert "minimaxi.com" in llm.openai_api_base
        assert llm.model_name == "MiniMax-M2.7"


def test_doctor_summary_structure():
    """doctor_summary 返回正确结构."""
    from shared.chatmodel_factory import doctor_summary

    s = doctor_summary()
    assert "providers" in s
    assert "frameworks" in s
    assert isinstance(s["providers"], list)
    assert isinstance(s["frameworks"], dict)
    for p in s["providers"]:
        assert "name" in p
        assert "display" in p
        assert "has_key" in p
        assert "region" in p
    assert "langchain" in s["frameworks"]
    assert "llama_index" in s["frameworks"]


# ═══════════════════════════════════════════════════════════
# env
# ═══════════════════════════════════════════════════════════


def test_get_api_key_known_providers():
    """get_api_key 已知厂商映射正确."""
    from shared.env import get_api_key

    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "sk-d",
            "KIMI_API_KEY": "sk-k",
            "SILICONFLOW_API_KEY": "sk-s",
            "MINIMAX_API_KEY": "sk-cp-x",
            "LLM_MOCK": "0",
        },
        clear=False,
    ):
        assert get_api_key("deepseek") == "sk-d"
        assert get_api_key("kimi") == "sk-k"
        assert get_api_key("siliconflow") == "sk-s"
        assert get_api_key("MiniMax") == "sk-cp-x"
        # 别名: moonshot → KIMI
        assert get_api_key("moonshot") == "sk-k"
        assert get_api_key("abab") == "sk-cp-x"  # MiniMax 旧模型 alias


def test_get_api_key_unknown_provider_fallback():
    """未知厂商: 转大写 + _API_KEY 拼接."""
    from shared.env import get_api_key

    with patch.dict(
        os.environ,
        {"FOO_BAR_API_KEY": "sk-fb", "LLM_MOCK": "0"},
        clear=False,
    ):
        assert get_api_key("foo_bar") == "sk-fb"


def test_mock_mode_does_not_expose_api_key_or_load_dotenv():
    """LLM_MOCK=1 不应读取 .env，也不应向调用方返回进程中的 Key。"""
    from shared import env as shared_env

    with (
        patch.dict(
            os.environ,
            {"LLM_MOCK": "1", "DEEPSEEK_API_KEY": "sk-test-must-stay-unused"},
            clear=False,
        ),
        patch.object(shared_env, "_find_dotenv") as find_dotenv,
    ):
        assert shared_env.get_api_key("deepseek") is None
        find_dotenv.assert_not_called()


def test_unset_mode_does_not_expose_process_key_or_load_dotenv():
    """未设置 LLM_MOCK 也是默认离线，不能使用进程中已有的 Key。"""
    from shared import env as shared_env

    with (
        patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "sk-test-must-stay-unused"},
            clear=True,
        ),
        patch.object(shared_env, "_find_dotenv") as find_dotenv,
    ):
        assert shared_env.get_api_key("deepseek") is None
        find_dotenv.assert_not_called()


def test_get_env():
    """get_env 简单包装."""
    from shared.env import get_env

    with patch.dict(os.environ, {"MY_KEY": "value"}, clear=False):
        assert get_env("MY_KEY") == "value"
        assert get_env("MY_KEY", "default") == "value"
        assert get_env("MISSING", "default") == "default"


print("OK")
