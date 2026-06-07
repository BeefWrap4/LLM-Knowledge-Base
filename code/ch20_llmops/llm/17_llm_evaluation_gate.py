# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.7.2 自动化评估门禁
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 17_llm_evaluation_gate.py
# expected_runtime: < 1s
# expected_output: Evaluation report dict with checks and recommendation
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2072-自动化评估门禁-⭐⭐⭐⭐
# Interview hooks:
#  - LLM 评估门禁和传统单元测试的本质区别是什么？
#  - 哪些指标适合作为硬门禁（CI fail），哪些适合作为软门禁（warning）？
#  - 评估集大小与统计稳定性如何权衡？

import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class QualityGate:
    """评估门禁配置"""
    min_accuracy: float = 0.85
    max_hallucination_rate: float = 0.05
    max_latency_p95_ms: float = 3000
    max_cost_per_query: float = 0.05
    require_safety_check: bool = True


class LLMEvaluationGate:
    """LLM 应用的自动化评估门禁"""

    def __init__(
        self,
        gate_config: QualityGate,
        eval_fn: Callable[..., Dict],
        test_dataset: List[Dict],
    ):
        self.config = gate_config
        self.eval_fn = eval_fn
        self.test_dataset = test_dataset

    def run_evaluation(self, prompt_version: str) -> Dict:
        results: List[Dict] = []
        total_cost = 0.0
        total_latency = 0.0
        for test_case in self.test_dataset:
            result = self.eval_fn(
                prompt_version=prompt_version,
                query=test_case["query"],
                expected=test_case.get("expected"),
                context=test_case.get("context"),
            )
            results.append(result)
            total_cost += result.get("cost", 0)
            total_latency += result.get("latency_ms", 0)

        n = len(results)
        avg_accuracy = sum(r.get("correct", 0) for r in results) / n
        hallucination_count = sum(r.get("hallucination", False) for r in results)
        avg_cost = total_cost / n
        p95_latency = sorted(r.get("latency_ms", 0) for r in results)[int(n * 0.95)]

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
                "✅ 所有门禁通过，可以部署"
                if all_passed
                else "❌ 门禁未通过！请检查失败项并修复"
            ),
        }

    def save_report(self, report: Dict, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    def mock_eval_fn(prompt_version, query, expected=None, context=None):
        return {
            "correct": random.random() > 0.1,
            "hallucination": random.random() < 0.03,
            "cost": random.uniform(0.001, 0.02),
            "latency_ms": random.gauss(800, 200),
        }

    gate = LLMEvaluationGate(
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
