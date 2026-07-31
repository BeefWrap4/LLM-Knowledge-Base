"""真实 API runner 与 CI 的 fail-closed 门禁；所有测试均禁止联网。"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent


def _load_script(name: str):
    path = CODE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_gate_test_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llm_doctor_default_is_network_free(monkeypatch) -> None:
    doctor = _load_script("llm_doctor")
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.setattr(
        doctor,
        "test_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应调用真实探针")),
    )
    assert doctor.main([]) == 0


def test_llm_doctor_requires_both_real_mode_and_confirmation(monkeypatch) -> None:
    doctor = _load_script("llm_doctor")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-only")
    monkeypatch.setattr(
        doctor,
        "test_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("门禁关闭时不得探针")),
    )

    monkeypatch.delenv("LLM_MOCK", raising=False)
    assert doctor.main(["--provider", "deepseek", "--confirm-real"]) == 2

    monkeypatch.setenv("LLM_MOCK", "0")
    assert doctor.main(["--provider", "deepseek"]) == 2


def test_llm_doctor_propagates_probe_failure(monkeypatch) -> None:
    doctor = _load_script("llm_doctor")
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-only")
    monkeypatch.setattr(
        doctor,
        "test_provider",
        lambda _name, api_key=None: (False, "", 0.01, "synthetic failure"),
    )

    assert doctor.main(["--provider", "deepseek", "--confirm-real"]) == 1


def test_llm_doctor_setup_uses_hidden_input_and_does_not_probe(monkeypatch, tmp_path) -> None:
    doctor = _load_script("llm_doctor")
    monkeypatch.setattr(doctor, "CODE", tmp_path)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "1")
    monkeypatch.setattr(doctor.getpass, "getpass", lambda _prompt: "sk-unit-hidden")
    monkeypatch.setattr(
        doctor,
        "test_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("setup 默认不得探针")),
    )

    assert doctor.main(["--setup"]) == 0
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "sk-unit-hidden" in env_text


def test_real_smoke_gate_refuses_incomplete_opt_in(monkeypatch) -> None:
    smoke = _load_script("test_real_api_smoke")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-only")
    monkeypatch.setattr(
        smoke,
        "run_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("门禁关闭时不得探针")),
    )

    monkeypatch.delenv("LLM_MOCK", raising=False)
    assert smoke.main(["--provider", "deepseek", "--confirm-real"]) == 2

    monkeypatch.setenv("LLM_MOCK", "0")
    assert smoke.main(["--provider", "deepseek"]) == 2


def test_real_smoke_strict_probe_uses_real_response_evidence(monkeypatch) -> None:
    smoke = _load_script("test_real_api_smoke")
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-only")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)

    class FakeClient:
        is_mock = False

        def __init__(self, **_kwargs):
            pass

        def chat(self, **_kwargs):
            return SimpleNamespace(
                mock=False,
                raw=object(),
                content="OK",
                usage={"total_tokens": 2},
                provider="deepseek",
                model="unit-model",
            )

    monkeypatch.setattr(smoke, "UnifiedClient", FakeClient)
    assert smoke.main(["--provider", "deepseek", "--confirm-real"]) == 0


def test_real_workflows_declare_exact_gates_and_never_skip_as_pass() -> None:
    workflow_paths = (
        REPO_ROOT / ".github" / "workflows" / "ci-llm-doctor.yml",
        REPO_ROOT / ".github" / "workflows" / "integration-test.yml",
    )
    for path in workflow_paths:
        text = path.read_text(encoding="utf-8")
        yaml.safe_load(text)
        assert 'LLM_MOCK: "0"' in text
        assert 'RUN_REAL_INTEGRATION: "1"' in text
        assert "LLM_PROVIDER:" in text
        assert "exit 0" not in text
        assert "USE_REAL_API" not in text
        assert "pull_request:" not in text
        assert "\n  push:" not in text

    doctor_workflow = workflow_paths[0].read_text(encoding="utf-8")
    assert "--provider \"$LLM_PROVIDER\" --confirm-real" in doctor_workflow


def test_make_real_and_download_targets_match_current_cli() -> None:
    text = (CODE_ROOT / "Makefile").read_text(encoding="utf-8")
    assert 'LLM_MOCK=0 LLM_PROVIDER="$(PROVIDER)"' in text
    assert "--confirm-real" in text
    assert "unset LLM_MOCK" not in text
    assert "USE_REAL_API" not in text

    assert "download_models.py --required-only" in text
    assert "download_models.py --llm-medium --confirm-large" in text
    assert "download_models.py --world-model --confirm-large" in text
    assert "download_models.py --reasoner" in text
    assert "download_models.py --edge-mlx" in text
    assert "download_models.py --edge-gguf" in text
    assert "download_models.py --default" not in text
    assert "download_models.py --tier" not in text
