# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.3.5 Token 用量追踪
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 08_token_tracker.py
# expected_runtime: < 1s
# expected_output: Cost per call + usage summary, with alert when threshold exceeded
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2035-token-用量追踪-⭐⭐⭐
# Interview hooks:
#  - 如何设计一个支持预算告警的 Token 追踪器？
#  - 输入/输出 Token 的成本差异为什么在 LLM 中很关键？
#  - 为什么需要在 trace 里同时记录 user_id 和时间窗口？

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TokenTracker:
    """Token 用量追踪器 —— 面试常考设计模式"""

    daily_budget: float = 50.0  # 每日预算 $50
    alert_threshold: float = 0.8  # 80% 时告警

    _daily_usage: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _monthly_usage: float = 0.0
    _model_pricing: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        }
    )

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str = "default",
    ) -> float:
        """记录一次 LLM 调用并返回本次调用成本"""
        pricing = self._model_pricing.get(model, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        today = time.strftime("%Y-%m-%d")
        self._daily_usage[today] += total_cost
        self._monthly_usage += total_cost

        if self._daily_usage[today] > self.daily_budget * self.alert_threshold:
            self._send_alert(
                f"⚠️ Token 用量已达日预算的 {self.alert_threshold * 100:.0f}% "
                f"(${self._daily_usage[today]:.2f}/${self.daily_budget:.2f})"
            )

        return total_cost

    def get_usage_summary(self) -> dict:
        """获取用量摘要"""
        today = time.strftime("%Y-%m-%d")
        return {
            "today": {"date": today, "cost": self._daily_usage[today]},
            "monthly_total": self._monthly_usage,
            "budget_remaining": self.daily_budget - self._daily_usage[today],
        }

    def _send_alert(self, message: str):
        """发送告警（可接入 Slack/钉钉/邮件）"""
        print(f"[ALERT] {message}")


if __name__ == "__main__":
    tracker = TokenTracker(daily_budget=100.0)

    # 模拟调用
    cost = tracker.track_call("gpt-4o", input_tokens=2000, output_tokens=500)
    print(f"本次调用成本: ${cost:.4f}")

    # 模拟大量调用触发告警
    for _ in range(100):
        tracker.track_call("gpt-4o", input_tokens=10000, output_tokens=2000)

    print(tracker.get_usage_summary())
