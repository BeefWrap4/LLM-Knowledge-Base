# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.10 Agent 回滚策略
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 25_agent_rollback_controller.py
# expected_runtime: < 1s
# expected_output: Rollback decision dicts printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#201010-agent-回滚策略agent-rollback
# Interview hooks:
#  - 为什么 Agent 的回滚需要 4 个层级（流量 / Prompt / Tool / Model）？
#  - 各级回滚的响应时间（SLO）应该如何排序？
#  - 成本超限自动回滚（1.5x budget）这个阈值如何避免抖动？

from enum import Enum

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "agent-rollback"}))
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)


class RollbackLevel(Enum):
    TRAFFIC_SHIFT = 1
    PROMPT_VERSION = 2
    TOOL_ALLOWLIST = 3
    MODEL_VERSION = 4


class AgentRollbackController:
    def __init__(self):
        self.tracer = trace.get_tracer("agent.rollback")

    def detect_rollback_signal(self, span: trace.Span, signal: str, severity: float):
        """检测回滚信号，写入 Span Event"""
        span.add_event(
            "rollback.signal",
            attributes={
                "rollback.signal.name": signal,
                "rollback.signal.severity": severity,
            },
        )

    def execute_rollback(
        self,
        level: RollbackLevel,
        reason: str,
        target_version: str | None = None,
    ) -> dict:
        """执行多层回滚"""
        with self.tracer.start_as_current_span(f"rollback.{level.name}") as span:
            span.set_attribute("rollback.level", level.value)
            span.set_attribute("rollback.reason", reason)
            span.set_attribute("rollback.target_version", target_version or "")

            if level == RollbackLevel.TRAFFIC_SHIFT:
                action = self._shift_traffic_to_old()
            elif level == RollbackLevel.PROMPT_VERSION:
                action = self._rollback_prompt_version(target_version)
            elif level == RollbackLevel.TOOL_ALLOWLIST:
                action = self._disable_risky_tools(["send_email", "execute_code", "delete_file"])
            elif level == RollbackLevel.MODEL_VERSION:
                action = self._rollback_model_version(target_version)
            else:
                action = "noop"
            span.set_attribute("rollback.action", action)
            return {"level": level.value, "action": action, "reason": reason}

    def _shift_traffic_to_old(self) -> str:
        return "k8s_traffic_shifted_to_v2.2.0"

    def _rollback_prompt_version(self, version: str) -> str:
        return f"prompt_registry_rollback_to_{version}"

    def _disable_risky_tools(self, tools) -> str:
        return f"tool_allowlist_disabled: {','.join(tools)}"

    def _rollback_model_version(self, version: str) -> str:
        return f"model_pinned_to_{version}"


def cost_overrun_auto_rollback(controller, current_cost_per_hour, budget_per_hour):
    """当 1h 成本超预算 1.5x 时自动触发回滚"""
    if current_cost_per_hour > budget_per_hour * 1.5:
        return controller.execute_rollback(
            level=RollbackLevel.TRAFFIC_SHIFT,
            reason=f"cost_overrun:{current_cost_per_hour:.2f}>{budget_per_hour:.2f}*1.5",
            target_version="v2.2.0",
        )
    return None


if __name__ == "__main__":
    controller = AgentRollbackController()

    # 模拟 4 种回滚场景
    print(
        controller.execute_rollback(RollbackLevel.TRAFFIC_SHIFT, "error_rate_5pct", target_version="v2.2.0")
    )
    print(
        controller.execute_rollback(
            RollbackLevel.PROMPT_VERSION, "hallucination_score_low", target_version="v3.2.0"
        )
    )
    print(controller.execute_rollback(RollbackLevel.TOOL_ALLOWLIST, "tool_error_rate_high"))
    print(
        controller.execute_rollback(
            RollbackLevel.MODEL_VERSION, "p99_latency_high", target_version="gpt-4o-2024-08"
        )
    )

    # 成本超限自动回滚
    auto = cost_overrun_auto_rollback(controller, current_cost_per_hour=15.0, budget_per_hour=10.0)
    print("auto:", auto)

    spans = exporter.get_finished_spans()
    print(f"spans emitted: {len(spans)}")
