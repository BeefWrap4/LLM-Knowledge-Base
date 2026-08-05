# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.token_tracker
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 08_token_tracker.py
# expected_runtime: < 1s
# expected_output: Cost per call + usage summary, with configurable alert threshold
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - 如何设计一个支持预算告警的 Token 追踪器？
#  - 输入/输出 Token 的成本差异为什么在 LLM 中很关键？
#  - 为什么 Rate Card 必须有来源、版本和生效时间？

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenRates:
    input_usd_per_million: float
    output_usd_per_million: float
    source: str


@dataclass
class TokenTracker:
    """Token 用量追踪器；预算和价格均由业务配置注入。"""

    model_rates: dict[str, TokenRates]
    daily_budget_usd: float
    alert_threshold_ratio: float = 0.8
    _daily_usage: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _monthly_usage: float = 0.0

    def __post_init__(self):
        if self.daily_budget_usd <= 0:
            raise ValueError("daily_budget_usd must be positive")
        if not 0 < self.alert_threshold_ratio <= 1:
            raise ValueError("alert_threshold_ratio must be in (0, 1]")

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str = "default",
    ) -> float:
        """记录一次调用；未知模型直接失败，避免静默套用错误价格。"""
        del user_id  # 真实系统应将用户维度写入受控标签；此示例不保留标识符。
        if model not in self.model_rates:
            raise KeyError(f"missing rate card for model={model!r}")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        rates = self.model_rates[model]
        total_cost = (
            input_tokens / 1_000_000 * rates.input_usd_per_million
            + output_tokens / 1_000_000 * rates.output_usd_per_million
        )

        today = time.strftime("%Y-%m-%d")
        self._daily_usage[today] += total_cost
        self._monthly_usage += total_cost
        if self._daily_usage[today] > self.daily_budget_usd * self.alert_threshold_ratio:
            self._send_alert(
                f"Token 成本已达教学预算阈值 {self.alert_threshold_ratio:.0%}: "
                f"${self._daily_usage[today]:.2f}/${self.daily_budget_usd:.2f}"
            )
        return total_cost

    def get_usage_summary(self) -> dict:
        today = time.strftime("%Y-%m-%d")
        return {
            "today": {"date": today, "cost_usd": self._daily_usage[today]},
            "monthly_total_usd": self._monthly_usage,
            "budget_remaining_usd": self.daily_budget_usd - self._daily_usage[today],
            "rate_sources": sorted({rate.source for rate in self.model_rates.values()}),
        }

    @staticmethod
    def _send_alert(message: str):
        print(f"[ALERT] {message}")


if __name__ == "__main__":
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
    # 下列默认费率仅用于演示计算；生产值必须来自供应商价格页/合同或实际账单。
    rates = TokenRates(
        input_usd_per_million=float(os.environ.get("LLM_INPUT_USD_PER_MILLION", "1")),
        output_usd_per_million=float(os.environ.get("LLM_OUTPUT_USD_PER_MILLION", "4")),
        source=os.environ.get("LLM_RATE_SOURCE", "illustrative-demo-rate-card"),
    )
    tracker = TokenTracker(
        model_rates={model: rates},
        daily_budget_usd=float(os.environ.get("LLM_DAILY_BUDGET_USD", "1")),
        alert_threshold_ratio=float(os.environ.get("LLM_BUDGET_ALERT_RATIO", "0.8")),
    )
    cost = tracker.track_call(model, input_tokens=2000, output_tokens=500)
    print(f"本次调用成本（教学费率）: ${cost:.6f}")
    tracker.track_call(model, input_tokens=800_000, output_tokens=100_000)
    print(tracker.get_usage_summary())
    print("OK")
