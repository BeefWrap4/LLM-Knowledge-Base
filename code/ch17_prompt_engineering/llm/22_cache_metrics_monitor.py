# ---
# chapter: 18
# topic: Context Engineering
# topic_id: prompt_engineering.cache_metrics_monitor
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 22_cache_metrics_monitor.py
# expected_runtime: <1s
# expected_output: 打印滑动窗口下的平均命中率与健康状态
# ---
# See: ../../../18_Context_Engineering.md
# Interview hooks:
# - 为何要使用滑动窗口而非全局累积统计？
# - 命中率告警阈值如何选择？(业务相关)
# - 如何把指标接入 Prometheus / OpenTelemetry？

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CacheMetrics:
    """接收已按供应商口径归一化的 cached/total input tokens。"""

    window_size: int = 100
    history: deque[tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError("window_size 必须大于 0")
        self.history = deque(maxlen=self.window_size)

    def record(self, cached_tokens: int, total_input_tokens: int):
        if not 0 <= cached_tokens <= total_input_tokens:
            raise ValueError("token 指标不合法或尚未按供应商口径归一化")
        self.history.append((cached_tokens, total_input_tokens))

    @property
    def weighted_reuse_rate(self) -> float:
        cached = sum(item[0] for item in self.history)
        total = sum(item[1] for item in self.history)
        return cached / total if total else 0.0

    def is_healthy(self, threshold: float) -> bool:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold 必须在 [0, 1]")
        return self.weighted_reuse_rate >= threshold


def send_alert(message: str):
    """模拟告警系统：生产中应接入飞书/钉钉/PagerDuty 等。"""
    print(f"[ALERT] {message}")


if __name__ == "__main__":
    metrics = CacheMetrics(window_size=10)
    # 模拟 10 次请求：前 5 次命中率高，后 5 次命中率低
    for i in range(5):
        metrics.record(cached_tokens=800, total_input_tokens=1000)
    print(f"[阶段1] 5 次高命中后 weighted_reuse_rate = {metrics.weighted_reuse_rate:.2%}")
    print(f"[健康?] {metrics.is_healthy(threshold=0.5)}")

    for i in range(5):
        metrics.record(cached_tokens=100, total_input_tokens=1000)
    print(f"\n[阶段2] 再 5 次低命中后 weighted_reuse_rate = {metrics.weighted_reuse_rate:.2%}")
    print(f"[健康?] {metrics.is_healthy(threshold=0.5)}")

    if not metrics.is_healthy(threshold=0.5):
        send_alert(f"缓存复用率低: {metrics.weighted_reuse_rate:.2%}")
    print("OK")
