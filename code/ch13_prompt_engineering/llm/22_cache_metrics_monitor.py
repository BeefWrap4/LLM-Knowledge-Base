# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.5 缓存命中率监控与告警
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 22_cache_metrics_monitor.py
# expected_runtime: <1s
# expected_output: 打印滑动窗口下的平均命中率与健康状态
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.5
# Interview hooks:
# - 为何要使用滑动窗口而非全局累积统计？
# - 命中率告警阈值如何选择？(业务相关)
# - 如何把指标接入 Prometheus / OpenTelemetry？

import time
from dataclasses import dataclass, field
from collections import deque


@dataclass
class CacheMetrics:
    """缓存指标监控"""
    window_size: int = 100
    cache_read_tokens: int = 0
    input_tokens: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, cache_read: int, new_input: int):
        self.cache_read_tokens += cache_read
        self.input_tokens += new_input
        hit_rate = cache_read / max(cache_read + new_input, 1)
        self.history.append({"timestamp": time.time(), "hit_rate": hit_rate})

    @property
    def avg_hit_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(h["hit_rate"] for h in self.history) / len(self.history)

    def is_healthy(self) -> bool:
        """命中率低于 50% 触发告警"""
        return self.avg_hit_rate >= 0.5


def send_alert(message: str):
    """模拟告警系统：生产中应接入飞书/钉钉/PagerDuty 等。"""
    print(f"[ALERT] {message}")


# 集成到 Anthropic 调用
def wrapped_call(messages, client=None, metrics: CacheMetrics = None, **kwargs):
    """演示包装函数：真实使用时 client 应为 anthropic.Anthropic()。"""
    # mock 一个响应
    class _MockUsage:
        cache_read_input_tokens = 800
        input_tokens = 200

    class _MockResp:
        usage = _MockUsage()
    response = _MockResp()

    metrics.record(
        cache_read=response.usage.cache_read_input_tokens,
        new_input=response.usage.input_tokens
    )
    if not metrics.is_healthy():
        send_alert(f"缓存命中率低: {metrics.avg_hit_rate:.2%}")
    return response


if __name__ == "__main__":
    metrics = CacheMetrics()
    # 模拟 10 次请求：前 5 次命中率高，后 5 次命中率低
    for i in range(5):
        metrics.record(cache_read=800, new_input=200)
    print(f"[阶段1] 5 次高命中后 avg_hit_rate = {metrics.avg_hit_rate:.2%}")
    print(f"[健康?] {metrics.is_healthy()}")

    for i in range(5):
        metrics.record(cache_read=100, new_input=900)
    print(f"\n[阶段2] 再 5 次低命中后 avg_hit_rate = {metrics.avg_hit_rate:.2%}")
    print(f"[健康?] {metrics.is_healthy()}")

    if not metrics.is_healthy():
        send_alert(f"缓存命中率低: {metrics.avg_hit_rate:.2%}")
    print("OK")
