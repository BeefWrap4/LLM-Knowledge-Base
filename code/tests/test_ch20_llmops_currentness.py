"""第 20 章的时效性、计费注入、OTel 语义与离线安全回归。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import util
from pathlib import Path
from types import ModuleType

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
CHAPTER = REPO_ROOT / "20_LLMOps与模型可观测性.md"
CHAPTER_ROOT = CODE_ROOT / "ch20_llmops"
EXAMPLE_ROOT = CHAPTER_ROOT / "llm"
SOURCE_LEDGER = REPO_ROOT / "docs" / "AUTHORITATIVE_SOURCES.md"

RETIRED_MODEL_IDS = (
    "gpt-4o",
    "gpt-4.5",
    "claude-sonnet-4",
    "claude-opus-4",
)
STALE_FIXED_CLAIMS = (
    "50-90%",
    "50%–90%",
    "30-60%",
    "30%–60%",
    "80-95%",
    "80%–95%",
)
OFFLINE_ENTRYPOINTS = (
    "02_mlflow_llm_tracking.py",
    "03_wandb_llm_tracking.py",
    "04_mlflow_hyperparam_search.py",
    "05_langsmith_observability.py",
    "06_langfuse_observability.py",
    "19_otel_genai_telemetry.py",
    "20_openinference_dual_semconv.py",
)


def load_example(filename: str) -> ModuleType:
    module_name = f"test_ch20_{Path(filename).stem}"
    spec = util.spec_from_file_location(module_name, EXAMPLE_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_ch20_does_not_pin_retired_model_ids_or_unsupported_savings_claims():
    paths = [CHAPTER, CHAPTER_ROOT / "README.md", *EXAMPLE_ROOT.glob("*.py")]
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(model_id in source for model_id in RETIRED_MODEL_IDS), path
        assert not any(claim.lower() in source for claim in STALE_FIXED_CLAIMS), path


def test_ch20_source_ledger_points_to_current_primary_sources():
    ledger = SOURCE_LEDGER.read_text(encoding="utf-8")
    row = next(line for line in ledger.splitlines() if line.startswith("| Ch20 |"))

    assert "developers.openai.com/api/docs/models" in row
    assert "openai.com/api/pricing" in row
    assert "platform.claude.com/docs/en/about-claude/models/overview" in row
    assert "platform.claude.com/docs/en/about-claude/pricing" in row
    assert "open-telemetry/semantic-conventions-genai" in row
    assert "mlflow.org/docs/latest/genai" in row
    assert "langfuse.com/docs/v4" in row
    assert "python-v3-to-v4" in row


def test_langfuse_example_uses_current_v4_api_and_disables_content_capture():
    example = (EXAMPLE_ROOT / "06_langfuse_observability.py").read_text(encoding="utf-8")
    combined = CHAPTER.read_text(encoding="utf-8") + example

    for current_api in (
        "from langfuse import get_client, observe, propagate_attributes",
        "update_current_span",
        "score_current_trace",
        "create_score",
        "capture_input=False",
        "capture_output=False",
    ):
        assert current_api in combined
    for retired_api in (
        "langfuse.decorators",
        "langfuse_context",
        "langfuse.score(",
        "update_current_trace(",
    ):
        assert retired_api not in combined


def test_current_otel_genai_fields_are_preserved_without_deprecated_custom_overloads():
    source = (EXAMPLE_ROOT / "19_otel_genai_telemetry.py").read_text(encoding="utf-8")
    required_fields = (
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.request.temperature",
        "gen_ai.request.max_tokens",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.cache_creation.input_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "gen_ai.client.token.usage",
        "gen_ai.token.type",
    )
    for field in required_fields:
        assert field in source

    chapter_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in [CHAPTER, *EXAMPLE_ROOT.glob("*.py")]
    )
    for deprecated_field in (
        "gen_ai.system",
        "gen_ai.usage.cached_input_tokens",
        "gen_ai.agent.trajectory_id",
        "gen_ai.cost.usd",
    ):
        assert deprecated_field not in chapter_sources


def test_cost_comparison_uses_the_injected_rate_card():
    module = load_example("01_cost_comparison.py")
    rates = module.TokenRates(
        input_usd_per_million=2.0,
        output_usd_per_million=7.0,
        source="unit-test-rate-card",
    )

    result = module.CostComparison.llm_cost(
        prompt_tokens_per_day=2_000_000,
        completion_tokens_per_day=1_000_000,
        model="test-model",
        rates=rates,
    )

    assert result["daily_cost"] == 11.0
    assert result["blended_cost_per_1k_input_tokens"] == 0.0055
    assert result["rate_source"] == "unit-test-rate-card"


def test_token_tracker_rejects_unknown_models_and_uses_injected_rates():
    module = load_example("08_token_tracker.py")
    tracker = module.TokenTracker(
        model_rates={
            "configured-model": module.TokenRates(
                input_usd_per_million=2.0,
                output_usd_per_million=6.0,
                source="unit-test-rate-card",
            )
        },
        daily_budget_usd=100.0,
    )

    assert tracker.track_call("configured-model", 1_000_000, 500_000) == 5.0
    with pytest.raises(KeyError):
        tracker.track_call("unknown-model", 1, 1)


def test_token_estimator_separates_token_estimation_from_injected_billing(monkeypatch):
    module = load_example("11_token_estimator.py")
    monkeypatch.setattr(
        module.TokenEstimator,
        "count_tokens",
        classmethod(lambda cls, text, model: 2_000_000),
    )
    config = module.ModelCostConfig(
        input_usd_per_million=3.0,
        output_usd_per_million=8.0,
        context_window_tokens=4_000_000,
        source="unit-test-rate-card",
    )

    result = module.TokenEstimator.estimate_cost(
        "content is irrelevant after monkeypatch",
        expected_output_tokens=500_000,
        model="test-model",
        config=config,
    )

    assert result["input_cost_usd_estimated"] == 6.0
    assert result["output_cost_usd_estimated"] == 4.0
    assert result["total_cost_usd_estimated"] == 10.0
    assert result["context_window_used_pct_estimated"] == 50.0


def test_exact_hash_cache_reports_lookup_denominator_and_injected_savings():
    module = load_example("12_semantic_cache.py")
    cache = module.ExactPromptCache()

    assert cache.get("prompt", "model", temperature=0.1) is None
    cache.set("prompt", "model", "response", temperature=0.1)
    assert cache.get("prompt", "model", temperature=0.1) == "response"
    assert cache.get("prompt", "model", temperature=0.7) is None

    assert cache.get_cache_stats() == {
        "cache_entries": 1,
        "lookups": 3,
        "total_hits": 1,
        "total_misses": 2,
        "hit_rate": pytest.approx(1 / 3),
    }
    assert cache.estimated_savings(0.25) == 0.25


def test_ab_framework_handles_missing_feedback_error_guardrail_and_json_export(tmp_path: Path):
    module = load_example("09_ab_test_framework.py")
    config = module.ABTestConfig(
        experiment_id="test",
        control_prompt="{question}",
        treatment_prompt="{question}",
        max_latency_ratio=2.0,
        max_token_ratio=2.0,
        max_error_rate_absolute_increase=0.1,
        minimum_relative_lift_pct=1.0,
        min_sample_size=1,
    )
    framework = module.LLMABTestFramework(config)
    framework.record_result(
        module.ABTestResult(
            user_id="control",
            variant=module.Variant.CONTROL,
            query="q",
            response="a",
            user_rated_helpful=True,
            request_succeeded=True,
            latency_ms=1,
            total_tokens=1,
        )
    )
    framework.record_result(
        module.ABTestResult(
            user_id="treatment",
            variant=module.Variant.TREATMENT,
            query="q",
            response="a",
            user_rated_helpful=False,
            request_succeeded=False,
            latency_ms=1,
            total_tokens=1,
        )
    )
    # Missing feedback is retained in raw traffic but excluded from the rated denominator.
    framework.record_result(
        module.ABTestResult(
            user_id="missing-feedback",
            variant=module.Variant.TREATMENT,
            query="q",
            response="a",
            user_rated_helpful=None,
            request_succeeded=True,
            latency_ms=1,
            total_tokens=1,
        )
    )

    analysis = framework.analyze()

    assert analysis["primary_metric"]["rated_sample_sizes"] == {
        "control": 1,
        "treatment": 1,
    }
    assert analysis["guardrail_metrics"]["error_rate"]["degraded"] is True

    output = tmp_path / "ab-results.json"
    framework.export_results(str(output))
    assert json.loads(output.read_text(encoding="utf-8"))[0]["variant"] == "control"


def test_metrics_collector_uses_nearest_rank_and_does_not_invent_a_time_window():
    module = load_example("13_llm_metrics_collector.py")
    collector = module.LLMMetricsCollector(
        error_rate_threshold=0.1,
        p95_latency_threshold_ms=200,
    )
    for latency in range(1, 101):
        collector.record_request("model", latency, 0, 0, 0.0)

    assert collector.get_latency_percentiles()["p95"] == 95
    empty = module.LLMMetricsCollector(
        error_rate_threshold=0.1,
        p95_latency_threshold_ms=200,
    )
    assert empty.check_alerts()[0]["message"] == "当前采集窗口内无任何请求"


def test_embedding_drift_reports_insufficient_data_and_uses_multiple_test_correction():
    module = load_example("14_embedding_drift_detector.py")
    detector = module.EmbeddingDriftDetector(
        lambda text: [float(len(text)), 1.0],
        centroid_distance_threshold=0.1,
        ks_familywise_alpha=0.05,
        reference_window_size=4,
        min_samples_per_window=2,
        max_ks_dimensions=2,
    )
    detector.add_reference("a")
    detector.add_current("b")

    assert detector.detect_drift()["drift_detected"] is None
    source = (EXAMPLE_ROOT / "14_embedding_drift_detector.py").read_text(encoding="utf-8")
    assert "Bonferroni" in source
    assert "avg_ks_pvalue" not in source
    detector.embed_fn = lambda text: [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        detector.add_current("dimension-change")


def test_evaluation_gate_is_fail_closed_for_missing_safety_results():
    module = load_example("17_llm_evaluation_gate.py")
    config = module.QualityGate(
        min_accuracy=0.5,
        max_hallucination_rate=0.5,
        max_latency_p95_ms=100,
        max_cost_per_query=1.0,
        require_safety_check=True,
    )
    gate = module.LLMEvaluationGate(
        config,
        lambda **kwargs: {
            "correct": True,
            "hallucination": False,
            "latency_ms": 10,
            "cost": 0.1,
        },
        [{"query": "q"}],
    )

    report = gate.run_evaluation("prompt")

    assert report["checks"]["safety"]["passed"] is False
    assert report["evaluation_passed"] is False
    with pytest.raises(ValueError):
        module.LLMEvaluationGate(config, lambda **kwargs: {}, []).run_evaluation("prompt")


def test_canary_rollback_sets_new_traffic_to_zero_and_resets_stage_window():
    module = load_example("18_canary_controller.py")
    controller = module.CanaryController(
        new_version="new",
        old_version="old",
        promotion_max_error_rate=0.1,
        rollback_error_rate=0.2,
        stage_min_minutes={
            module.ReleaseStage.CANARY_5: 0,
            module.ReleaseStage.CANARY_25: 0,
            module.ReleaseStage.CANARY_50: 0,
        },
        min_health_checks_per_stage=1,
    )
    assert controller.get_error_rate() is None
    controller.record_health_check(True)
    assert controller.should_promote() is True
    controller.promote()
    assert controller.health_checks_total == 0

    controller.rollback("quality regression")

    assert controller.get_traffic_split() == 0
    assert controller.get_status()["rollback_reason"] == "quality regression"


def test_offline_entrypoint_sources_do_not_read_provider_keys_directly():
    key_names = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "WANDB_API_KEY",
    )
    for filename in OFFLINE_ENTRYPOINTS:
        source = (EXAMPLE_ROOT / filename).read_text(encoding="utf-8")
        assert not any(key_name in source for key_name in key_names), filename


@pytest.mark.parametrize("filename", OFFLINE_ENTRYPOINTS)
def test_llm_mock_overrides_live_opt_in_and_runs_without_network(filename: str, tmp_path: Path):
    script = EXAMPLE_ROOT / filename
    env = os.environ.copy()
    env.update(
        {
            "LLM_MOCK": "1",
            "LLM_REAL_API": "1",
            "MLFLOW_TRACKING_URI": "https://127.0.0.1:1",
            "OPENAI_API_KEY": "fake-key-that-must-not-be-used",
            "ANTHROPIC_API_KEY": "fake-key-that-must-not-be-used",
            "LANGSMITH_API_KEY": "fake-key-that-must-not-be-used",
            "LANGFUSE_SECRET_KEY": "fake-key-that-must-not-be-used",
            "WANDB_API_KEY": "fake-key-that-must-not-be-used",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "",
        }
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.parametrize("filename", OFFLINE_ENTRYPOINTS)
def test_unset_mock_mode_stays_offline_even_with_live_opt_in(filename: str, tmp_path: Path):
    """LLM_REAL_API 不是主门禁；未设置 LLM_MOCK 时仍不得联网。"""
    script = EXAMPLE_ROOT / filename
    env = os.environ.copy()
    env.pop("LLM_MOCK", None)
    env.update(
        {
            "LLM_REAL_API": "1",
            "MLFLOW_TRACKING_URI": "https://127.0.0.1:1",
            "OPENAI_API_KEY": "fake-key-that-must-not-be-used",
            "LANGSMITH_API_KEY": "fake-key-that-must-not-be-used",
            "LANGFUSE_SECRET_KEY": "fake-key-that-must-not-be-used",
            "WANDB_API_KEY": "fake-key-that-must-not-be-used",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "",
        }
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
