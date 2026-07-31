"""运行文档的静态一致性门禁。"""

from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = CODE_ROOT / "docs"
DOC_NAMES = (
    "MIGRATE_LANGCHAIN.md",
    "MIGRATE_TO_UNIFIED.md",
    "REAL_DEMOS.md",
    "API_KEYS.md",
    "DEPLOY_LOCAL.md",
)
DOC_PATHS = tuple(DOC_ROOT / name for name in DOC_NAMES)


@pytest.fixture(scope="module")
def docs() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in DOC_PATHS}


def test_runtime_docs_exist_and_define_explicit_modes(docs: dict[str, str]) -> None:
    assert set(docs) == set(DOC_NAMES)
    for name, text in docs.items():
        assert "LLM_MOCK=1" in text, f"{name} 缺离线模式说明"
        assert "LLM_MOCK=0" in text, f"{name} 缺真实模式说明"


def test_runtime_docs_remove_stale_operational_claims(docs: dict[str, str]) -> None:
    combined = "\n".join(docs.values())
    stale_claims = (
        "Wave ",
        "USE_REAL_API",
        "真实实测通过",
        "自动降级 mock",
        "全厂商兼容",
        "0 行业务改造",
        "100% 通过",
        "注册送 2000 万",
        "新用户 ¥15",
        "无限调用",
    )
    for claim in stale_claims:
        assert claim not in combined


def test_provider_snapshot_matches_registry(docs: dict[str, str]) -> None:
    from shared.provider_registry import PROVIDERS

    api_keys_doc = docs["API_KEYS.md"]
    target_names = {
        "deepseek",
        "kimi",
        "siliconflow",
        "MiniMax",
        "openai",
        "anthropic",
    }
    providers = {provider.name: provider for provider in PROVIDERS.values()}
    assert target_names <= providers.keys()

    for name in sorted(target_names):
        provider = providers[name]
        expected_values = (
            provider.name,
            provider.env_key,
            provider.default_chat,
            provider.base_url,
        )
        for value in expected_values:
            assert f"`{value}`" in api_keys_doc, f"API_KEYS.md 未同步 {name}: {value}"
        if provider.default_reasoner:
            assert f"`{provider.default_reasoner}`" in api_keys_doc


def test_real_commands_explicitly_disable_mock(docs: dict[str, str]) -> None:
    for name, text in docs.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            is_real_runner = (
                "bash scripts/run_real_demos.sh" in line
                or "python scripts/test_integration.py" in line
            )
            is_provider_python = "LLM_PROVIDER=" in line and "python " in line
            if is_real_runner or is_provider_python:
                has_real_flag = "LLM_MOCK=0" in line or '$env:LLM_MOCK="0"' in line
                assert has_real_flag, f"{name}:{lineno} 真实命令未显式 LLM_MOCK=0"
            if "bash scripts/run_real_demos.sh" in line:
                assert "--confirm-real" in line, f"{name}:{lineno} demo runner 缺确认标志"
            if "python scripts/test_integration.py" in line:
                assert "RUN_REAL_INTEGRATION=1" in line, f"{name}:{lineno} 集成测试缺确认标志"


def test_legacy_cloud_models_are_absent_or_historical(docs: dict[str, str]) -> None:
    combined = "\n".join(docs.values())
    for retired in (
        "deepseek-chat",
        "deepseek-reasoner",
        "MiniMax-Text-01",
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ):
        assert retired not in combined

    historical_examples = {
        "MIGRATE_LANGCHAIN.md": "gpt-4o-mini",
        "MIGRATE_TO_UNIFIED.md": "gpt-4",
    }
    for name, model in historical_examples.items():
        text = docs[name]
        history_marker = text.index("历史 Before")
        assert history_marker < text.index(model)


def test_docs_describe_real_acceptance_boundary(docs: dict[str, str]) -> None:
    assert "assert not r.mock" in docs["MIGRATE_TO_UNIFIED.md"]
    assert "quick_chat()" in docs["MIGRATE_TO_UNIFIED.md"]
    assert "assert not r.mock" in docs["API_KEYS.md"]
    assert "wrapper 不是“真实通过”门禁" in docs["REAL_DEMOS.md"]
    assert "条件性组件 smoke" in docs["DEPLOY_LOCAL.md"]
