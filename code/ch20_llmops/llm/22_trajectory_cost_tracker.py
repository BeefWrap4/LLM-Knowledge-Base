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

import hashlib
import os
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
        query_lower = user_query.lower()
        query_category = "coding" if "python" in query_lower or "代码" in user_query else "general"
        with self.tracer.start_as_current_span(
            f"trajectory.{agent_name}",
            attributes={
                "gen_ai.agent.name": agent_name,
                "app.agent.trajectory_id": trajectory_id,
                # Keep raw content out of default telemetry. Hashes are still pseudonymous data.
                "app.agent.query_length": len(user_query),
                "app.agent.query_category": query_category,
                "app.agent.query_hash": hashlib.sha256(user_query.encode("utf-8")).hexdigest()[:16],
            },
        ) as root:
            cost_attrs = {"app.agent.trajectory_id": trajectory_id}
            try:
                yield trajectory_id, cost_attrs
                root.set_status(Status(StatusCode.OK))
            except Exception as e:
                root.set_status(Status(StatusCode.ERROR, str(e)))
                root.record_exception(e)
                raise

    def attribute_subspan(self, span, trajectory_id: str):
        span.set_attribute("app.agent.trajectory_id", trajectory_id)

    def cost_summary(self, trajectory_id: str, spans) -> dict:
        total_cost = 0.0
        total_input = 0
        total_output = 0
        total_reasoning = 0
        total_input_cost = 0.0
        total_output_cost = 0.0
        llm_calls = 0
        tool_calls = 0
        for s in spans:
            if s.attributes.get("app.agent.trajectory_id") != trajectory_id:
                continue
            if s.attributes.get("openinference.span.kind") == "LLM":
                llm_calls += 1
                total_cost += float(s.attributes.get("app.llm.cost.usd", 0))
                total_input += int(s.attributes.get("gen_ai.usage.input_tokens", 0))
                total_output += int(s.attributes.get("gen_ai.usage.output_tokens", 0))
                total_reasoning += int(s.attributes.get("gen_ai.usage.reasoning.output_tokens", 0))
                total_input_cost += float(s.attributes.get("app.llm.cost.input.usd", 0))
                total_output_cost += float(s.attributes.get("app.llm.cost.output.usd", 0))
            elif s.attributes.get("openinference.span.kind") == "TOOL":
                tool_calls += 1
        return {
            "trajectory_id": trajectory_id,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_reasoning_tokens": total_reasoning,
            "observed_cost_breakdown_usd": {
                "input": round(total_input_cost, 6),
                "output_including_reasoning": round(total_output_cost, 6),
                "unattributed": round(max(total_cost - total_input_cost - total_output_cost, 0), 6),
            },
        }


if __name__ == "__main__":
    tracker = TrajectoryCostTracker()
    tracer = trace.get_tracer("agent.demo")
    balanced_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    fast_model = os.environ.get("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5")
    # 成本值模拟“供应商响应/账单已归因结果”，不是写死的模型价格。
    observed_input_cost = float(os.environ.get("DEMO_OBSERVED_INPUT_COST_USD", "0.002"))
    observed_output_cost = float(os.environ.get("DEMO_OBSERVED_OUTPUT_COST_USD", "0.003"))

    with tracker.trajectory("帮我写一个 Python 装饰器") as (traj_id, _cost_attrs):
        # 模拟 2 次 LLM + 1 次 TOOL
        for kind, model in [
            ("LLM", balanced_model),
            ("TOOL", None),
            ("LLM", fast_model),
        ]:
            with tracer.start_as_current_span(f"{kind}.{model or 'tool'}") as sp:
                tracker.attribute_subspan(sp, traj_id)
                sp.set_attribute("openinference.span.kind", kind)
                if kind == "LLM":
                    sp.set_attribute("app.llm.cost.input.usd", observed_input_cost)
                    sp.set_attribute("app.llm.cost.output.usd", observed_output_cost)
                    sp.set_attribute("app.llm.cost.usd", observed_input_cost + observed_output_cost)
                    sp.set_attribute("gen_ai.usage.input_tokens", 1000)
                    sp.set_attribute("gen_ai.usage.output_tokens", 200)
                    sp.set_attribute("gen_ai.usage.reasoning.output_tokens", 100)

    summary = tracker.cost_summary(traj_id, exporter.get_finished_spans())
    print(summary)
    print("OK")
