"""Ch13 LLM examples must be offline unless real API access is explicitly enabled."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROOT = CODE_ROOT / "ch17_prompt_engineering" / "llm"
CASES = [
    ("12_anthropic_extended_thinking.py", "call_anthropic", ("test",), "anthropic", "ANTHROPIC_API_KEY"),
    (
        "13_anthropic_prompt_caching.py",
        "call_anthropic_with_cache",
        ([], []),
        "anthropic",
        "ANTHROPIC_API_KEY",
    ),
    ("14_openai_auto_caching.py", "call_openai", ([],), "openai", "OPENAI_API_KEY"),
    ("15_gemini_explicit_caching.py", "run_gemini_cache_demo", (), "gemini", "GEMINI_API_KEY"),
    ("16_claude_computer_use.py", "call_claude_computer_use", ("test",), "anthropic", "ANTHROPIC_API_KEY"),
    ("17_openai_cua.py", "call_openai_computer", ("test",), "openai", "OPENAI_API_KEY"),
    ("20_openai_json_schema_strict.py", "call_openai_structured", ("test",), "openai", "OPENAI_API_KEY"),
]


def _load_example(filename: str):
    path = EXAMPLE_ROOT / filename
    module_name = f"_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fail_client_construction(*_args, **_kwargs):
    raise AssertionError("offline path must not construct an SDK client")


def _replace_client(module, provider: str) -> None:
    if provider == "anthropic":
        module.anthropic = SimpleNamespace(Anthropic=_fail_client_construction)
    elif provider == "openai":
        module.OpenAI = _fail_client_construction
    else:
        module.genai = SimpleNamespace(Client=_fail_client_construction)
        module.types = SimpleNamespace()


@pytest.mark.parametrize(
    ("filename", "entrypoint", "args", "provider", "key_name"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_default_mode_does_not_construct_client(
    monkeypatch,
    capsys,
    filename: str,
    entrypoint: str,
    args: tuple,
    provider: str,
    key_name: str,
):
    module = _load_example(filename)
    _replace_client(module, provider)
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.setenv(key_name, "fake-key-that-must-not-be-used")

    result = getattr(module, entrypoint)(*args)

    output = capsys.readouterr().out
    assert result is None
    assert "[SKIP]" in output
    assert "OK" in output


@pytest.mark.parametrize(
    ("filename", "entrypoint", "args", "provider", "key_name"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_explicit_real_mode_still_requires_key(
    monkeypatch,
    capsys,
    filename: str,
    entrypoint: str,
    args: tuple,
    provider: str,
    key_name: str,
):
    module = _load_example(filename)
    _replace_client(module, provider)
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.delenv(key_name, raising=False)

    result = getattr(module, entrypoint)(*args)

    output = capsys.readouterr().out
    assert result is None
    assert "[SKIP]" in output
    assert "OK" in output


@pytest.mark.parametrize("filename", [case[0] for case in CASES])
def test_main_is_offline_with_fake_keys(filename: str):
    env = os.environ.copy()
    env.update(
        {
            "LLM_MOCK": "1",
            "ANTHROPIC_API_KEY": "fake-anthropic-key",
            "OPENAI_API_KEY": "fake-openai-key",
            "GEMINI_API_KEY": "fake-gemini-key",
        }
    )
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / filename)],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "[SKIP]" in result.stdout
    assert "OK" in result.stdout
