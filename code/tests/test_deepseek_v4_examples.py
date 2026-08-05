"""DeepSeek V4 provider configuration and offline example regressions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    CODE_ROOT / "shared" / "provider_registry.py",
    CODE_ROOT / "ch30_lora_qlora" / "gpu" / "08_deepseek_edge.py",
    CODE_ROOT / "ch30_lora_qlora" / "gpu" / "11_adaptive_inference.py",
    CODE_ROOT / "ch32_reasoning_ttc" / "llm" / "04_reasoning_effort_ladder.py",
    CODE_ROOT / "ch32_reasoning_ttc" / "llm" / "07_s1_budget_forcing.py",
    CODE_ROOT / "ch32_reasoning_ttc" / "llm" / "08_s1_wait_token.py",
]
LLM_EXAMPLES = TARGETS[3:]


def test_deepseek_provider_uses_current_v4_models():
    from shared.provider_registry import get_provider

    provider = get_provider("deepseek")
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.default_chat == "deepseek-v4-flash"
    assert provider.default_reasoner == "deepseek-v4-pro"


def test_deepseek_targets_do_not_reference_retired_api_names():
    retired = ("deepseek-chat", "deepseek-reasoner", "https://api.deepseek.com/v1")
    for path in TARGETS:
        source = path.read_text(encoding="utf-8")
        assert not any(value in source for value in retired), path


def test_deepseek_thinking_examples_use_official_controls():
    thinking_examples = [TARGETS[1], *LLM_EXAMPLES]
    for path in thinking_examples:
        source = path.read_text(encoding="utf-8")
        assert 'reasoning_effort="high"' in source or 'reasoning_effort=effort' in source
        assert 'extra_body={"thinking": {"type": "enabled"}}' in source


def test_deepseek_edge_mock_skips_before_key_or_network():
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [sys.executable, str(TARGETS[1]), "--mock"],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "[SKIP]" in result.stdout
    assert "OK" in result.stdout


@pytest.mark.parametrize("script", LLM_EXAMPLES, ids=lambda path: path.name)
def test_deepseek_llm_examples_have_offline_path(script: Path):
    env = os.environ.copy()
    env["LLM_MOCK"] = "1"
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "deepseek-v4-pro" in result.stdout


@pytest.mark.parametrize("script", LLM_EXAMPLES, ids=lambda path: path.name)
def test_deepseek_llm_examples_default_to_offline_even_if_key_exists(script: Path):
    env = os.environ.copy()
    env.pop("LLM_MOCK", None)
    env["DEEPSEEK_API_KEY"] = "sk-test-must-not-be-used"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "离线演示" in result.stdout
