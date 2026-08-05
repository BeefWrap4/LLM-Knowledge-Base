# ---
# chapter: 44
# topic: LLMOps 生命周期与持续交付
# topic_id: llmops.llm_evaluation_gate
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 17_llm_evaluation_gate.py
# expected_runtime: < 1s
# expected_output: Evaluation report dict with checks and recommendation
# ---
# See: ../../../44_LLMOps生命周期与持续交付.md
# Interview hooks:
#  - LLM 评估门禁和传统单元测试的本质区别是什么？
#  - 哪些指标适合作为硬门禁（CI fail），哪些适合作为软门禁（warning）？
#  - 评估集大小与统计稳定性如何权衡？

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityGate:
    """业务注入的评估门禁；不存在跨任务通用默认阈值。"""

    min_accuracy: float
    max_hallucination_rate: float
    max_latency_p95_ms: float
    max_cost_per_query: float
    require_safety_check: bool = True

    def __post_init__(self):
        if not 0 <= self.min_accuracy <= 1:
            raise ValueError("min_accuracy must be in [0, 1]")
        if not 0 <= self.max_hallucination_rate <= 1:
            raise ValueError("max_hallucination_rate must be in [0, 1]")
        if self.max_latency_p95_ms <= 0 or self.max_cost_per_query < 0:
            raise ValueError("latency threshold must be positive and cost budget non-negative")


class LLMEvaluationGate:
    """LLM 应用的自动化评估门禁"""

    def __init__(
        self,
        gate_config: QualityGate,
        eval_fn: Callable[..., dict],
        test_dataset: list[dict],
    ):
        self.config = gate_config
        self.eval_fn = eval_fn
        self.test_dataset = test_dataset

    def run_evaluation(self, prompt_version: str) -> dict:
        if not self.test_dataset:
            raise ValueError("test_dataset must not be empty")
        results: list[dict] = []
        total_cost = 0.0
        for test_case in self.test_dataset:
            result = self.eval_fn(
                prompt_version=prompt_version,
                query=test_case["query"],
                expected=test_case.get("expected"),
                context=test_case.get("context"),
            )
            results.append(result)
            observed_cost = result.get("cost", 0)
            if observed_cost < 0:
                raise ValueError("observed cost must be non-negative")
            total_cost += observed_cost

        n = len(results)
        avg_accuracy = sum(r.get("correct", 0) for r in results) / n
        hallucination_count = sum(r.get("hallucination", False) for r in results)
        avg_cost = total_cost / n
        latencies = sorted(r.get("latency_ms", 0) for r in results)
        if any(latency < 0 for latency in latencies):
            raise ValueError("latency_ms must be non-negative")
        p95_latency = latencies[max(0, math.ceil(0.95 * n) - 1)]

        checks = {
            "accuracy": {
                "value": avg_accuracy,
                "threshold": self.config.min_accuracy,
                "passed": avg_accuracy >= self.config.min_accuracy,
            },
            "hallucination_rate": {
                "value": hallucination_count / n,
                "threshold": self.config.max_hallucination_rate,
                "passed": (hallucination_count / n) <= self.config.max_hallucination_rate,
            },
            "latency_p95_ms": {
                "value": p95_latency,
                "threshold": self.config.max_latency_p95_ms,
                "passed": p95_latency <= self.config.max_latency_p95_ms,
            },
            "cost_per_query": {
                "value": avg_cost,
                "threshold": self.config.max_cost_per_query,
                "passed": avg_cost <= self.config.max_cost_per_query,
            },
        }
        if self.config.require_safety_check:
            safety_passed = all(bool(result.get("safety_passed", False)) for result in results)
            checks["safety"] = {
                "value": safety_passed,
                "threshold": True,
                # 缺失安全结果按失败处理，避免 fail-open。
                "passed": safety_passed,
            }
        all_passed = all(c["passed"] for c in checks.values())
        return {
            "prompt_version": prompt_version,
            "dataset_size": n,
            "evaluation_passed": all_passed,
            "checks": checks,
            "details": {
                "test_cases": len(self.test_dataset),
                "total_cost": round(total_cost, 4),
                "avg_accuracy": round(avg_accuracy, 3),
            },
            "recommendation": (
                "✅ 所有门禁通过，可以部署" if all_passed else "❌ 门禁未通过！请检查失败项并修复"
            ),
        }

    def save_report(self, report: dict, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    random.seed(0)

    def mock_eval_fn(prompt_version, query, expected=None, context=None):
        return {
            "correct": random.random() > 0.1,
            "hallucination": random.random() < 0.03,
            "cost": random.uniform(0.001, 0.02),
            "latency_ms": max(0, random.gauss(800, 200)),
            "safety_passed": True,
        }

    gate = LLMEvaluationGate(
        # 以下仅是教学策略；生产值由标注集基线、统计不确定性、SLO 和预算确定。
        gate_config=QualityGate(
            min_accuracy=0.85,
            max_hallucination_rate=0.05,
            max_latency_p95_ms=3000,
            max_cost_per_query=0.05,
        ),
        eval_fn=mock_eval_fn,
        test_dataset=[{"query": f"test query {i}"} for i in range(50)],
    )
    report = gate.run_evaluation("prompt_v4.0.0")
    print(report["recommendation"])
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    print("OK")
