"""Offline acceptance tests for Ch24 external-service and evaluation gates."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent.parent
GPU = CODE / "ch43_cloudnative" / "gpu"


def _run(script: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    if env:
        child_env.update(env)
    return subprocess.run(
        [sys.executable, str(GPU / script), *args],
        cwd=CODE,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.core
@pytest.mark.parametrize(
    "script",
    [
        "01_fastapi_vllm_server.py",
        "02_grpc_vllm_server.py",
        "04_eval_gate.py",
        "05_ollama_cloud_integration.py",
        "06_sagemaker_deploy.py",
    ],
)
def test_ch24_gpu_mock_mode_skips_before_real_side_effects(script: str) -> None:
    result = _run(script, "--mock")
    assert result.returncode == 0, result.stderr
    assert "[SKIP]" in result.stdout
    assert result.stdout.rstrip().endswith("OK")


@pytest.mark.core
def test_eval_gate_computes_metrics_from_scored_jsonl(tmp_path: Path) -> None:
    records = [
        {"correct": i != 0, "safe": True, "latency_ms": 100 + i, "token_error": False}
        for i in range(10)
    ]
    dataset = tmp_path / "scored.jsonl"
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "results.json"

    result = _run(
        "04_eval_gate.py",
        "--eval-dataset",
        str(dataset),
        "--threshold-accuracy",
        "0.9",
        "--threshold-safety",
        "1",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert "EVALUATION GATE PASSED" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accuracy"] == pytest.approx(0.9)
    assert payload["safety_score"] == pytest.approx(1.0)
    assert payload["latency_p99_ms"] == pytest.approx(109)
    assert payload["total_test_cases"] == 10
    assert payload["source_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert payload["gate"]["passed"] is True


@pytest.mark.core
def test_eval_gate_rejects_unscored_or_incomplete_records(tmp_path: Path) -> None:
    dataset = tmp_path / "raw-prompts.jsonl"
    dataset.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    output = tmp_path / "results.json"

    result = _run(
        "04_eval_gate.py",
        "--eval-dataset",
        str(dataset),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert "评估输入无效" in result.stderr
    assert not output.exists()


@pytest.mark.core
def test_sagemaker_defaults_to_no_sdk_no_request_dry_run() -> None:
    result = _run("06_sagemaker_deploy.py", env={"SAGEMAKER_DEPLOY": "0"})
    assert result.returncode == 0, result.stderr
    assert "DRY RUN ONLY" in result.stdout
    assert "未导入 AWS SDK、未读取凭证、未发请求、未创建资源" in result.stdout
    assert "[BILLING ACTIVE]" not in result.stdout


@pytest.mark.core
def test_sagemaker_deploy_refuses_without_independent_cost_gates() -> None:
    result = _run(
        "06_sagemaker_deploy.py",
        "--deploy",
        "--model-id",
        "owner/model",
        "--role-arn",
        "arn:aws:iam::123456789012:role/tutorial",
        "--region",
        "us-east-1",
        "--endpoint-name",
        "tutorial-endpoint",
        "--instance-type",
        "ml.g5.xlarge",
        "--transformers-version",
        "validated-version",
        "--pytorch-version",
        "validated-version",
        "--py-version",
        "py311",
        "--keep-endpoint",
        env={"SAGEMAKER_DEPLOY": "0"},
    )
    assert result.returncode == 2
    assert "SAGEMAKER_DEPLOY=1" in result.stderr
    assert "confirm-deploy" in result.stderr
    assert "SageMaker deploy failed" not in result.stderr


@pytest.mark.core
def test_ollama_cloud_refuses_before_import_or_request_without_confirmation() -> None:
    result = _run(
        "05_ollama_cloud_integration.py",
        "--mode",
        "cloud",
        "--cloud-model",
        "current-model-from-tags",
        env={"OLLAMA_API_KEY": "not-used-by-this-test"},
    )
    assert result.returncode == 2
    assert "--confirm-cloud" in result.stderr
    assert "Ollama request failed" not in result.stderr


@pytest.mark.core
def test_real_paths_do_not_contain_silent_mock_engines() -> None:
    banned = (
        "MockVLLMEngine",
        "MockAsyncEngine",
        "_MockOllamaClient",
        "_MockHuggingFaceModel",
        "using mock SDK",
    )
    for script in (
        "01_fastapi_vllm_server.py",
        "02_grpc_vllm_server.py",
        "05_ollama_cloud_integration.py",
        "06_sagemaker_deploy.py",
    ):
        source = (GPU / script).read_text(encoding="utf-8")
        for marker in banned:
            assert marker not in source, f"{script} still contains {marker}"


@pytest.mark.core
def test_grpc_example_ships_the_proto_it_tells_users_to_compile() -> None:
    proto = GPU / "protos" / "llm_inference.proto"
    source = proto.read_text(encoding="utf-8")
    assert "service LLMInference" in source
    assert "rpc GenerateStream" in source
