"""Ch17 examples use current APIs and remain offline unless real mode is explicit."""

import builtins
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
EXAMPLE_ROOT = CODE_ROOT / "ch17_evaluation" / "llm"
CHAPTER = REPO_ROOT / "17_大模型评估体系.md"
SOURCE_LEDGER = REPO_ROOT / "docs" / "AUTHORITATIVE_SOURCES.md"

OPTIONAL_IMPORT_ROOTS = {
    "openai",
    "ragas",
    "trulens",
    "deepeval",
    "langfuse",
    "phoenix",
    "pandas",
    "bert_score",
    "torch",
    "transformers",
}

OFFLINE_CASES = [
    ("03_bertscore_metric.py", "compute_bertscore_demo", (), RuntimeError),
    ("04_perplexity.py", "compute_perplexity", ("test",), RuntimeError),
    ("05_llm_as_judge.py", "offline_judge", (), dict),
    ("06_ragas_evaluation.py", "run_ragas_evaluation", (), list),
    ("07_trulens_rag_triad.py", "setup_trulens_rag_triad", (), list),
    ("08_deepeval_rag.py", "run_deepeval_test", (), list),
    ("12_langfuse_v3.py", "run_langfuse_experiment_demo", (), type(None)),
    ("13_phoenix_auto_instrument.py", "run_phoenix_demo", (), type(None)),
    ("14_deepeval_dag_geval.py", "run_deepeval_dag_geval", (), list),
]

REAL_MODE_CASES = [
    ("05_llm_as_judge.py", "real_judge"),
    ("06_ragas_evaluation.py", "run_ragas_evaluation"),
    ("07_trulens_rag_triad.py", "setup_trulens_rag_triad"),
    ("08_deepeval_rag.py", "run_deepeval_test"),
    ("12_langfuse_v3.py", "run_langfuse_experiment_demo"),
    ("13_phoenix_auto_instrument.py", "run_phoenix_demo"),
    ("14_deepeval_dag_geval.py", "run_deepeval_dag_geval"),
]


def _load_example(filename: str) -> Any:
    path = EXAMPLE_ROOT / filename
    module_name = f"_test_ch17_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_entrypoint(module: Any, name: str, args: tuple[Any, ...]) -> Any:
    if name == "offline_judge":
        return module.LLMJudge().evaluate("question", "answer")
    if name == "real_judge":
        return module.LLMJudge()
    return getattr(module, name)(*args)


@pytest.mark.parametrize(
    ("filename", "entrypoint", "args", "expected_type"),
    OFFLINE_CASES,
    ids=[case[0] for case in OFFLINE_CASES],
)
def test_default_mode_does_not_import_network_stacks(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    entrypoint: str,
    args: tuple[Any, ...],
    expected_type: type[Any],
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *import_args: Any, **import_kwargs: Any) -> Any:
        if name.partition(".")[0] in OPTIONAL_IMPORT_ROOTS:
            raise AssertionError(f"offline path imported optional network/model stack: {name}")
        return original_import(name, *import_args, **import_kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-that-must-not-be-read")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-key-that-must-not-be-read")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-key-that-must-not-be-read")

    module = _load_example(filename)
    if expected_type is RuntimeError:
        with pytest.raises(RuntimeError, match="LLM_MOCK=1"):
            _call_entrypoint(module, entrypoint, args)
    else:
        result = _call_entrypoint(module, entrypoint, args)
        assert isinstance(result, expected_type)


@pytest.mark.parametrize(
    ("filename", "entrypoint"),
    REAL_MODE_CASES,
    ids=[case[0] for case in REAL_MODE_CASES],
)
def test_explicit_real_mode_requires_credentials_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    entrypoint: str,
) -> None:
    monkeypatch.setenv("LLM_MOCK", "0")
    for variable in (
        "OPENAI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    module = _load_example(filename)
    with pytest.raises(RuntimeError, match="真实模式"):
        _call_entrypoint(module, entrypoint, ())


def test_judge_uses_responses_structured_output_and_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example("05_llm_as_judge.py")
    calls: list[dict[str, Any]] = []

    class FakeResponses:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "overall_score": 4,
                        "dimensions": {
                            "accuracy": 4,
                            "completeness": 4,
                            "clarity": 4,
                            "helpfulness": 4,
                        },
                        "strengths": ["clear"],
                        "weaknesses": [],
                        "justification": "calibrated fixture",
                    }
                )
            )

    monkeypatch.setenv("LLM_MOCK", "0")
    judge = module.LLMJudge(client=SimpleNamespace(responses=FakeResponses()))
    result = judge.evaluate("question", "answer")

    assert result["overall_score"] == 4
    assert calls[0]["model"] == "gpt-5.6"
    assert calls[0]["reasoning"] == {"effort": "low"}
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True
    assert "temperature" not in calls[0]

    class FailingResponses:
        def create(self, **_kwargs: Any) -> Any:
            raise ConnectionError("fixture failure")

    failing_judge = module.LLMJudge(client=SimpleNamespace(responses=FailingResponses()))
    with pytest.raises(ConnectionError, match="fixture failure"):
        failing_judge.evaluate("question", "answer")
    assert failing_judge.mock_mode is False


@pytest.mark.parametrize(
    "filename",
    sorted(path.name for path in EXAMPLE_ROOT.glob("*.py")),
)
def test_each_ch17_script_finishes_offline(filename: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "LLM_MOCK": "1",
            "OPENAI_API_KEY": "fake-key-that-must-not-be-used",
            "LANGFUSE_PUBLIC_KEY": "fake-key-that-must-not-be-used",
            "LANGFUSE_SECRET_KEY": "fake-key-that-must-not-be-used",
        }
    )
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / filename)],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_ch17_sources_and_current_api_markers() -> None:
    scripts = "\n".join(path.read_text(encoding="utf-8") for path in sorted(EXAMPLE_ROOT.glob("*.py")))
    chapter = CHAPTER.read_text(encoding="utf-8")
    ledger = SOURCE_LEDGER.read_text(encoding="utf-8")

    assert len(list(EXAMPLE_ROOT.glob("*.py"))) == 14
    assert "responses.create(" in scripts
    assert "from ragas.metrics.collections import" in scripts
    assert "from trulens.core import Metric, Selector" in scripts
    assert "get_client" in scripts and "run_experiment" in scripts
    assert "from phoenix.evals import LLM, evaluate_dataframe" in scripts
    assert "BinaryJudgementNode" in scripts and "SingleTurnParams" in scripts

    for stale_import in (
        "from ragas import evaluate",
        "from trulens_eval",
        "from langfuse.evaluation",
        "from phoenix.evals import run_evals",
        "LLMTestCaseParams",
        "TaskCompletionIndicator",
    ):
        assert stale_import not in scripts

    for retired_model_id in ("gpt-4o", "gpt-4-turbo", "gpt-3.5"):
        assert retired_model_id not in scripts

    assert "独立同分布（i.i.d.）" in chapter
    assert "$0.95^{10}\\approx 0.599$" in chapter
    assert "lm-eval run" in chapter
    assert "权重不由 OpenAI API 或 ChatGPT 托管" in chapter
    assert "| Ch17 |" in ledger
    assert "migrate_from_v03_to_v04" in ledger
    assert "experiments-via-sdk" in ledger
