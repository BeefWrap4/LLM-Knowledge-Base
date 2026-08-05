# ---
# chapter: 44
# topic: LLMOps 生命周期与持续交付
# topic_id: llmops.ab_test_framework
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 09_ab_test_framework.py
# expected_runtime: < 1s
# expected_output: Analysis dict with sample sizes, primary metric, guardrail metrics, recommendation
# ---
# See: ../../../44_LLMOps生命周期与持续交付.md
# Interview hooks:
#  - 为什么 A/B 测试要用 user_id 哈希做确定性分配，而不是随机数？
#  - Guardrail 指标（如延迟、Token 用量）为什么必须与主指标联合观察？
#  - 简化 Z-test 在二分类指标上如何判断显著性？

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from statistics import NormalDist
from typing import Any


class Variant(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class ABTestConfig:
    """A/B 测试配置"""

    experiment_id: str
    control_prompt: str
    treatment_prompt: str
    max_latency_ratio: float
    max_token_ratio: float
    max_error_rate_absolute_increase: float
    minimum_relative_lift_pct: float
    traffic_split: float = 0.5
    min_sample_size: int = 100
    significance_alpha: float = 0.05
    primary_metric: str = "user_satisfaction"
    guardrail_metrics: list[str] = field(
        default_factory=lambda: [
            "response_latency_ms",
            "token_usage",
            "error_rate",
        ]
    )
    status: str = "draft"

    def __post_init__(self):
        if not 0 < self.traffic_split < 1:
            raise ValueError("traffic_split must be in (0, 1)")
        if self.min_sample_size <= 0:
            raise ValueError("min_sample_size must be positive")
        if self.max_latency_ratio <= 0 or self.max_token_ratio <= 0:
            raise ValueError("guardrail ratios must be positive")
        if self.max_error_rate_absolute_increase < 0:
            raise ValueError("max_error_rate_absolute_increase must be non-negative")
        if self.minimum_relative_lift_pct < 0:
            raise ValueError("minimum_relative_lift_pct must be non-negative")
        if not 0 < self.significance_alpha < 1:
            raise ValueError("significance_alpha must be in (0, 1)")


@dataclass
class ABTestResult:
    """单次 A/B 测试结果"""

    user_id: str
    variant: Variant
    query: str
    response: str
    user_rated_helpful: bool | None = None
    user_clicked_source: bool | None = None
    conversation_continued: bool | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_succeeded: bool = True
    hallucination_detected: bool = False
    safety_flag_raised: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMABTestFramework:
    """LLM Prompt A/B 测试框架"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.results: list[ABTestResult] = []

    def assign_variant(self, user_id: str, query: str) -> Variant:
        """基于 user_id 哈希的确定性流量分配。"""
        del query  # 实验单位是用户；不能让同一用户因 query 改变分组。
        hash_input = f"{self.config.experiment_id}:{user_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        if bucket < self.config.traffic_split:
            return Variant.TREATMENT
        return Variant.CONTROL

    def get_prompt(self, variant: Variant, **kwargs) -> str:
        template = (
            self.config.treatment_prompt if variant == Variant.TREATMENT else self.config.control_prompt
        )
        return template.format(**kwargs)

    def record_result(self, result: ABTestResult):
        if result.latency_ms < 0 or result.total_tokens < 0:
            raise ValueError("latency_ms and total_tokens must be non-negative")
        self.results.append(result)

    def analyze(self) -> dict[str, Any]:
        control_results = [r for r in self.results if r.variant == Variant.CONTROL]
        treatment_results = [r for r in self.results if r.variant == Variant.TREATMENT]

        if len(control_results) < self.config.min_sample_size:
            return {"status": "insufficient_data", "message": "Control 组样本不足"}
        if len(treatment_results) < self.config.min_sample_size:
            return {"status": "insufficient_data", "message": "Treatment 组样本不足"}

        analysis: dict[str, Any] = {
            "experiment_id": self.config.experiment_id,
            "status": "analyzed",
            "sample_sizes": {
                "control": len(control_results),
                "treatment": len(treatment_results),
                "total": len(self.results),
            },
        }

        control_rated = [r for r in control_results if r.user_rated_helpful is not None]
        treatment_rated = [r for r in treatment_results if r.user_rated_helpful is not None]
        if len(control_rated) < self.config.min_sample_size:
            return {"status": "insufficient_data", "message": "Control 组有效评分样本不足"}
        if len(treatment_rated) < self.config.min_sample_size:
            return {"status": "insufficient_data", "message": "Treatment 组有效评分样本不足"}

        control_helpful = sum(r.user_rated_helpful is True for r in control_rated)
        treatment_helpful = sum(r.user_rated_helpful is True for r in treatment_rated)
        n_c = len(control_rated)
        n_t = len(treatment_rated)
        control_rate = control_helpful / n_c
        treatment_rate = treatment_helpful / n_t
        if control_rate > 0:
            relative_change = (treatment_rate - control_rate) / control_rate * 100
        else:
            relative_change = float("inf") if treatment_rate > 0 else 0.0

        analysis["primary_metric"] = {
            "name": "user_satisfaction",
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "relative_change": relative_change,
            "rated_sample_sizes": {"control": n_c, "treatment": n_t},
        }

        # 简化 Z-test
        p_pool = (control_helpful + treatment_helpful) / (n_c + n_t)
        if p_pool > 0 and p_pool < 1:
            se = (p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)) ** 0.5
            z_score = (treatment_rate - control_rate) / se if se > 0 else 0
            p_value = 2 * (1 - NormalDist().cdf(abs(z_score)))
            analysis["primary_metric"]["z_score"] = z_score
            analysis["primary_metric"]["p_value"] = p_value
            analysis["primary_metric"]["significant"] = (
                p_value < self.config.significance_alpha
            )
        else:
            analysis["primary_metric"]["p_value"] = 1.0
            analysis["primary_metric"]["significant"] = False

        # Guardrail 指标
        guardrails: dict[str, Any] = {}
        for metric in self.config.guardrail_metrics:
            if metric == "response_latency_ms":
                c_val = sum(r.latency_ms for r in control_results) / n_c
                t_val = sum(r.latency_ms for r in treatment_results) / n_t
                guardrails[metric] = {
                    "control": c_val,
                    "treatment": t_val,
                    "change_pct": (
                        (t_val - c_val) / c_val * 100
                        if c_val
                        else (float("inf") if t_val else 0.0)
                    ),
                    "degraded": t_val > c_val * self.config.max_latency_ratio,
                }
            elif metric == "token_usage":
                c_val = sum(r.total_tokens for r in control_results) / n_c
                t_val = sum(r.total_tokens for r in treatment_results) / n_t
                guardrails[metric] = {
                    "control": c_val,
                    "treatment": t_val,
                    "change_pct": (
                        (t_val - c_val) / c_val * 100
                        if c_val
                        else (float("inf") if t_val else 0.0)
                    ),
                    "degraded": t_val > c_val * self.config.max_token_ratio,
                }
            elif metric == "error_rate":
                c_val = sum(not r.request_succeeded for r in control_results) / len(control_results)
                t_val = sum(not r.request_succeeded for r in treatment_results) / len(
                    treatment_results
                )
                guardrails[metric] = {
                    "control": c_val,
                    "treatment": t_val,
                    "absolute_change": t_val - c_val,
                    "degraded": (
                        t_val - c_val > self.config.max_error_rate_absolute_increase
                    ),
                }
        analysis["guardrail_metrics"] = guardrails

        # 决策
        sig = analysis["primary_metric"].get("significant", False)
        rel_change = analysis["primary_metric"]["relative_change"]
        has_degradation = any(g.get("degraded", False) for g in guardrails.values())
        if sig and rel_change > self.config.minimum_relative_lift_pct and not has_degradation:
            analysis["recommendation"] = "✅ 建议上线 Treatment（统计显著正向提升，无明显劣化）"
        elif sig and rel_change < -self.config.minimum_relative_lift_pct:
            analysis["recommendation"] = "❌ Treatment 显著劣于 Control，建议放弃"
        elif has_degradation:
            analysis["recommendation"] = "⚠️ Guardrail 指标劣化，需进一步分析"
        else:
            analysis["recommendation"] = "⏳ 统计不显著，建议继续收集数据"
        return analysis

    def export_results(self, filepath: str):
        serialized = []
        for result in self.results:
            row = asdict(result)
            row["variant"] = result.variant.value
            serialized.append(row)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    random.seed(0)
    config = ABTestConfig(
        experiment_id="prompt-v4-beta1-vs-v3.2",
        control_prompt="你是一个问答助手。基于以下参考资料回答问题：\n{context}\n\n问题：{question}",
        treatment_prompt=(
            "你是一个专业问答助手。请先分析问题，再基于参考资料逐步推理，"
            "最后给出带来源标注的回答。\n\n参考资料：{context}\n\n问题：{question}\n\n"
            "请按以下格式回答：\n1. 分析：\n2. 回答：\n3. 来源："
        ),
        # 教学策略参数；生产值应在功效分析、历史基线与业务 SLO 上校准。
        max_latency_ratio=1.2,
        max_token_ratio=1.5,
        max_error_rate_absolute_increase=0.005,
        minimum_relative_lift_pct=5.0,
        traffic_split=0.5,
        min_sample_size=100,
    )
    framework = LLMABTestFramework(config)

    user_ids = [f"user_{i}" for i in range(500)]
    for uid in user_ids:
        variant = framework.assign_variant(uid, "test query")
        result = ABTestResult(
            user_id=uid,
            variant=variant,
            query="什么是 Python 装饰器？",
            response="A decorator is...",
            user_rated_helpful=random.random() < (0.7 if variant == Variant.TREATMENT else 0.6),
            latency_ms=random.gauss(800, 100),
            total_tokens=random.randint(300, 600),
            request_succeeded=random.random() > 0.002,
        )
        framework.record_result(result)

    analysis = framework.analyze()
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print("OK")
