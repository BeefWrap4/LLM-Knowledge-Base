# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.7 Per-Trajectory Cost Attribution
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 22_trajectory_cost_tracker.py
# expected_runtime: < 1s
# expected_output: Trajectory cost summary dict printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20107-per-trajectory-cost-attribution按轨迹成本归因
# Interview hooks:
#  - Agent 应用为何必须做"按 trajectory 归因"而不是按 Span？
#  - trajectory_id 在子 Span 间如何传播（contextvars / Baggage）？
#  - cost_breakdown 拆分到 input/output/thinking 的业务价值？

import uuid
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

provider = TracerProvider(resource=Resource.create({"service.name": "agent.trajectory"}))
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)


class TrajectoryCostTracker:
    def __init__(self):
        self.tracer = trace.get_tracer("agent.trajectory")

    @contextmanager
    def trajectory(self, user_query: str, agent_name: str = "react-agent"):
        trajectory_id = f"traj-{uuid.uuid4().hex[:12]}"
        with self.tracer.start_as_current_span(
            f"trajectory.{agent_name}",
            attributes={
                "gen_ai.agent.name": agent_name,
                "gen_ai.agent.trajectory_id": trajectory_id,
                "gen_ai.agent.user_query": user_query[:256],
            },
        ) as root:
            cost_attrs = {"gen_ai.agent.trajectory_id": trajectory_id}
            try:
                yield trajectory_id, cost_attrs
                root.set_status(Status(StatusCode.OK))
            except Exception as e:
                root.set_status(Status(StatusCode.ERROR, str(e)))
                root.record_exception(e)
                raise

    def attribute_subspan(self, span, trajectory_id: str):
        span.set_attribute("gen_ai.agent.trajectory_id", trajectory_id)

    def cost_summary(self, trajectory_id: str, spans) -> dict:
        total_cost = 0.0
        total_input = 0
        total_output = 0
        total_thinking = 0
        llm_calls = 0
        tool_calls = 0
        for s in spans:
            if s.attributes.get("gen_ai.agent.trajectory_id") != trajectory_id:
                continue
            if s.attributes.get("openinference.span.kind") == "LLM":
                llm_calls += 1
                total_cost += float(s.attributes.get("gen_ai.cost.usd", 0))
                total_input += int(s.attributes.get("gen_ai.usage.input_tokens", 0))
                total_output += int(s.attributes.get("gen_ai.usage.output_tokens", 0))
                total_thinking += int(s.attributes.get("gen_ai.thinking.tokens_used", 0))
            elif s.attributes.get("openinference.span.kind") == "TOOL":
                tool_calls += 1
        return {
            "trajectory_id": trajectory_id,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_thinking_tokens": total_thinking,
            "cost_breakdown": {
                "input": round(total_input / 1e6 * 3.0, 6),
                "output": round(total_output / 1e6 * 15.0, 6),
                "thinking": round(total_thinking / 1e6 * 15.0, 6),
            },
        }


if __name__ == "__main__":
    tracker = TrajectoryCostTracker()
    tracer = trace.get_tracer("agent.demo")

    with tracker.trajectory("帮我写一个 Python 装饰器") as (traj_id, _cost_attrs):
        # 模拟 2 次 LLM + 1 次 TOOL
        for kind, model in [
            ("LLM", "claude-sonnet-4-6"),
            ("TOOL", None),
            ("LLM", "claude-haiku-4-5"),
        ]:
            with tracer.start_as_current_span(f"{kind}.{model or 'tool'}") as sp:
                tracker.attribute_subspan(sp, traj_id)
                sp.set_attribute("openinference.span.kind", kind)
                if kind == "LLM":
                    sp.set_attribute("gen_ai.cost.usd", 0.005)
                    sp.set_attribute("gen_ai.usage.input_tokens", 1000)
                    sp.set_attribute("gen_ai.usage.output_tokens", 200)
                    sp.set_attribute("gen_ai.thinking.tokens_used", 100)
                sp.set_attribute("gen_ai.agent.trajectory_id", traj_id)

    summary = tracker.cost_summary(traj_id, exporter.get_finished_spans())
    print(summary)
