# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.llm_metrics_collector
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 13_llm_metrics_collector.py
# expected_runtime: < 1s
# expected_output: Latency percentiles, error rate, alert list printed
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - P50/P95/P99 延迟在 LLM 推理监控中为什么必须分桶？
#  - 错误率与 P95 延迟阈值如何从业务 SLO、历史基线和错误预算推导？
#  - 多线程场景下指标收集器为什么需要 Lock？

import math
import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class LLMMetricsCollector:
    """LLM 应用指标收集器 —— 面试中展示系统设计能力"""

    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    _total_requests: int = 0
    _successful_requests: int = 0
    _failed_requests: int = 0
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0
    _total_cost: float = 0.0
    _model_stats: dict = field(
        default_factory=lambda: defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0, "errors": 0})
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)
    error_rate_threshold: float = 0.01
    p95_latency_threshold_ms: float = 3000

    def __post_init__(self):
        if not 0 <= self.error_rate_threshold <= 1:
            raise ValueError("error_rate_threshold must be in [0, 1]")
        if self.p95_latency_threshold_ms <= 0:
            raise ValueError("p95_latency_threshold_ms must be positive")

    def record_request(
        self,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        success: bool = True,
    ):
        """记录一次请求的指标"""
        if latency_ms < 0 or input_tokens < 0 or output_tokens < 0 or cost < 0:
            raise ValueError("latency, token counts, and observed cost must be non-negative")
        with self._lock:
            self._latencies.append(latency_ms)
            self._total_requests += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost += cost

            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

            stats = self._model_stats[model]
            stats["requests"] += 1
            stats["tokens"] += input_tokens + output_tokens
            stats["cost"] += cost
            if not success:
                stats["errors"] += 1

    def get_latency_percentiles(self) -> dict[str, float]:
        """用 nearest-rank 计算当前有界窗口的经验分位数。"""
        with self._lock:
            latencies = list(self._latencies)
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        def nearest_rank(percentile: float) -> float:
            return sorted_lat[max(0, math.ceil(percentile * n) - 1)]

        return {
            "p50": nearest_rank(0.50),
            "p95": nearest_rank(0.95),
            "p99": nearest_rank(0.99),
            "avg": sum(sorted_lat) / n,
            "min": sorted_lat[0],
            "max": sorted_lat[-1],
        }

    def get_error_rate(self) -> float:
        with self._lock:
            if self._total_requests == 0:
                return 0.0
            return self._failed_requests / self._total_requests

    def get_summary(self) -> dict:
        with self._lock:
            total_requests = self._total_requests
            successful_requests = self._successful_requests
            failed_requests = self._failed_requests
            total_input_tokens = self._total_input_tokens
            total_output_tokens = self._total_output_tokens
            total_cost = self._total_cost
            model_stats = {model: dict(stats) for model, stats in self._model_stats.items()}
        return {
            "requests": {
                "total": total_requests,
                "successful": successful_requests,
                "failed": failed_requests,
                "error_rate": failed_requests / total_requests if total_requests else 0.0,
            },
            "latency": self.get_latency_percentiles(),
            "tokens": {
                "total_input": total_input_tokens,
                "total_output": total_output_tokens,
            },
            "cost": {"observed_total": round(total_cost, 4)},
            "per_model": model_stats,
        }

    def check_alerts(self) -> list[dict]:
        alerts: list[dict] = []
        summary = self.get_summary()
        if summary["requests"]["error_rate"] > self.error_rate_threshold:
            alerts.append(
                {
                    "severity": "critical",
                    "message": f"错误率过高: {summary['requests']['error_rate']:.2%}",
                    "threshold": f"> {self.error_rate_threshold:.2%}",
                }
            )
        if summary["latency"]["p95"] > self.p95_latency_threshold_ms:
            alerts.append(
                {
                    "severity": "warning",
                    "message": f"P95 延迟过高: {summary['latency']['p95']:.0f}ms",
                    "threshold": f"> {self.p95_latency_threshold_ms:.0f}ms",
                }
            )
        if summary["requests"]["total"] == 0:
            alerts.append(
                {
                    "severity": "critical",
                    "message": "当前采集窗口内无任何请求",
                }
            )
        return alerts


if __name__ == "__main__":
    import random

    random.seed(0)
    collector = LLMMetricsCollector(
        error_rate_threshold=float(os.environ.get("LLM_ERROR_RATE_THRESHOLD", "0.01")),
        p95_latency_threshold_ms=float(os.environ.get("LLM_P95_LATENCY_THRESHOLD_MS", "3000")),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
    for _ in range(200):
        collector.record_request(
            model=model,
            latency_ms=max(0, random.gauss(900, 250)),
            input_tokens=random.randint(100, 2000),
            output_tokens=random.randint(50, 500),
            cost=random.uniform(0.001, 0.05),
            success=random.random() > 0.005,
        )
    print("summary:", collector.get_summary())
    print("alerts:", collector.check_alerts())
    print("OK")
