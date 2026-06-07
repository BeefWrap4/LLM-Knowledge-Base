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
import pytest
from pathlib import Path
from unittest.mock import patch

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


def test_get_provider_unknown_returns_mock():
    """未知厂商名返回 mock."""
    from shared.provider_registry import get_provider, PROVIDERS
    p = get_provider("nonexistent_vendor_xyz")
    assert p.name == "mock"


def test_get_provider_known():
    """已知厂商名返回正确 Provider."""
    from shared.provider_registry import get_provider
    assert get_provider("deepseek").name == "deepseek"
    assert get_provider("MiniMax").name == "MiniMax"
    assert get_provider("kimi").display_name == "Kimi (月之暗面)"


def test_get_default_provider_no_key_returns_mock():
    """无 API Key 时默认厂商是 mock."""
    from shared.provider_registry import get_default_provider
    # 清空所有可能存在的 Key
    with patch.dict(os.environ, {}, clear=True):
        assert get_default_provider().name == "mock"


def test_get_default_provider_with_deepseek_key():
    """设 DEEPSEEK_API_KEY + LLM_PROVIDER=deepseek 时默认厂商是 deepseek."""
    from shared.provider_registry import get_default_provider
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "sk-test-xxx",
        "LLM_PROVIDER": "deepseek",
    }, clear=False):
        assert get_default_provider().name == "deepseek"


def test_get_default_provider_respects_env_override():
    """LLM_PROVIDER 环境变量优先级最高."""
    from shared.provider_registry import get_default_provider
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "sk-1",
        "KIMI_API_KEY": "sk-2",
        "LLM_PROVIDER": "kimi",
    }, clear=False):
        assert get_default_provider().name == "kimi"


# ═══════════════════════════════════════════════════════════
# llm_client
# ═══════════════════════════════════════════════════════════

def test_unified_client_no_key_falls_back_to_mock():
    """无 API Key 时 UnifiedClient 降级 mock."""
    from shared.llm_client import UnifiedClient
    with patch.dict(os.environ, {}, clear=True):
        c = UnifiedClient(provider="deepseek")
        assert c.is_mock is True
        resp = c.chat(prompt="test")
        assert resp.mock is True
        assert "确定性" in resp.content or "mock" in resp.content.lower() or len(resp.content) > 0


def test_unified_client_with_key_not_mock():
    """有 API Key 时 UnifiedClient 不降级."""
    from shared.llm_client import UnifiedClient
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-xxx"}, clear=False):
        c = UnifiedClient(provider="deepseek")
        assert c.is_mock is False
        assert c.api_key == "sk-test-xxx"
        assert c.client is not None


def test_unified_client_response_has_attrs():
    """_LLMResponse 应有 content / model / provider / mock 字段."""
    from shared.llm_client import UnifiedClient
    with patch.dict(os.environ, {}, clear=True):
        c = UnifiedClient()
        resp = c.chat(prompt="hello")
        assert hasattr(resp, "content")
        assert hasattr(resp, "model")
        assert hasattr(resp, "provider")
        assert hasattr(resp, "mock")
        assert hasattr(resp, "usage")


def test_unified_client_chat_with_messages():
    """messages 形式调用应工作."""
    from shared.llm_client import UnifiedClient
    with patch.dict(os.environ, {}, clear=True):
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


def test_make_chat_model_unknown_framework_raises():
    """未知 framework 抛 ValueError."""
    from shared.chatmodel_factory import make_chat_model
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
        with pytest.raises(ValueError, match="未知 framework"):
            make_chat_model(provider="deepseek", framework="unknown_fw")


def test_make_chat_model_default_provider():
    """不指定 provider + 显式 LLM_PROVIDER=deepseek 时用 deepseek."""
    from shared.chatmodel_factory import make_chat_model
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "sk-test",
        "LLM_PROVIDER": "deepseek",
    }, clear=False):
        llm = make_chat_model()
        # langchain ChatOpenAI 实例
        from langchain_openai import ChatOpenAI
        assert isinstance(llm, ChatOpenAI)
        assert "deepseek" in llm.openai_api_base


def test_make_chat_model_minimax():
    """MiniMax provider."""
    from shared.chatmodel_factory import make_chat_model
    from langchain_openai import ChatOpenAI
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-cp-test"}, clear=False):
        llm = make_chat_model(provider="MiniMax")
        assert isinstance(llm, ChatOpenAI)
        assert "minimaxi.com" in llm.openai_api_base
        assert llm.model_name == "MiniMax-Text-01"


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
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "sk-d",
        "KIMI_API_KEY": "sk-k",
        "SILICONFLOW_API_KEY": "sk-s",
        "MINIMAX_API_KEY": "sk-cp-x",
    }, clear=False):
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
    with patch.dict(os.environ, {"FOO_BAR_API_KEY": "sk-fb"}, clear=False):
        assert get_api_key("foo_bar") == "sk-fb"


def test_get_env():
    """get_env 简单包装."""
    from shared.env import get_env
    with patch.dict(os.environ, {"MY_KEY": "value"}, clear=False):
        assert get_env("MY_KEY") == "value"
        assert get_env("MY_KEY", "default") == "value"
        assert get_env("MISSING", "default") == "default"

print("OK")
