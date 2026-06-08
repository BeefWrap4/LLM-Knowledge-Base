# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.6.1 核心监控指标
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 13_llm_metrics_collector.py
# expected_runtime: < 1s
# expected_output: Latency percentiles, error rate, alert list printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2061-核心监控指标
# Interview hooks:
#  - P50/P95/P99 延迟在 LLM 推理监控中为什么必须分桶？
#  - 错误率告警阈值（1%）和 P95 延迟告警阈值（3s）是怎么定出来的？
#  - 多线程场景下指标收集器为什么需要 Lock？

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
        """计算延迟分位数"""
        if not self._latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return {
            "p50": sorted_lat[int(n * 0.50)],
            "p95": sorted_lat[int(n * 0.95)],
            "p99": sorted_lat[int(n * 0.99)],
            "avg": sum(sorted_lat) / n,
            "min": sorted_lat[0],
            "max": sorted_lat[-1],
        }

    def get_error_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._failed_requests / self._total_requests

    def get_summary(self) -> dict:
        return {
            "requests": {
                "total": self._total_requests,
                "successful": self._successful_requests,
                "failed": self._failed_requests,
                "error_rate": self.get_error_rate(),
            },
            "latency": self.get_latency_percentiles(),
            "tokens": {
                "total_input": self._total_input_tokens,
                "total_output": self._total_output_tokens,
            },
            "cost": {"total": round(self._total_cost, 4)},
            "per_model": dict(self._model_stats),
        }

    def check_alerts(self) -> list[dict]:
        alerts: list[dict] = []
        summary = self.get_summary()
        if summary["requests"]["error_rate"] > 0.01:
            alerts.append(
                {
                    "severity": "critical",
                    "message": f"错误率过高: {summary['requests']['error_rate']:.2%}",
                    "threshold": "> 1%",
                }
            )
        if summary["latency"]["p95"] > 3000:
            alerts.append(
                {
                    "severity": "warning",
                    "message": f"P95 延迟过高: {summary['latency']['p95']:.0f}ms",
                    "threshold": "> 3000ms",
                }
            )
        if self._total_requests == 0:
            alerts.append(
                {
                    "severity": "critical",
                    "message": "5 分钟内无任何请求",
                }
            )
        return alerts


if __name__ == "__main__":
    import random

    collector = LLMMetricsCollector()
    for _ in range(200):
        collector.record_request(
            model="gpt-4o-mini",
            latency_ms=random.gauss(900, 250),
            input_tokens=random.randint(100, 2000),
            output_tokens=random.randint(50, 500),
            cost=random.uniform(0.001, 0.05),
            success=random.random() > 0.005,
        )
    print("summary:", collector.get_summary())
    print("alerts:", collector.check_alerts())
