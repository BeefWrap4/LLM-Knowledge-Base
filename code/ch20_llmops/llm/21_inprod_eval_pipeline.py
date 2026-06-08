# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.10.5 in-prod Eval Pipeline 模式
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 21_inprod_eval_pipeline.py
# expected_runtime: < 1s
# expected_output: Span list with judge evaluation events and bad-case queue contents
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#20105-in-prod-eval-pipeline-模式
# Interview hooks:
#  - 为什么 in-prod eval 比离线评估更值得做？成本/样本偏差如何权衡？
#  - 1% 采样跑 Judge 的成本估算公式？
#  - bad case 自动入训练集的反馈回路有哪些工程陷阱（标签噪声、时序）？

import queue
import random

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "in-prod-eval"}))
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

JUDGE_PROBABILITY = 0.01  # 1% 流量跑 Judge
tracer = trace.get_tracer("in-prod-eval")
bad_case_queue: "queue.Queue" = queue.Queue()


def judge_relevance(query, response):
    return random.uniform(0.6, 0.99)


def judge_hallucination(response, ground_truth):
    return random.uniform(0.4, 0.95)


def judge_helpfulness(query, response):
    return random.uniform(0.5, 0.95)


def with_judge(llm_call_span, response_text: str, query: str, ground_truth=None):
    """在线上 Span 上挂载 Judge 评估（mocked 1% 采样）"""
    if random.random() > JUDGE_PROBABILITY:
        return None  # 采样外，跳过

    scores = {
        "relevance": judge_relevance(query, response_text),
        "hallucination": judge_hallucination(response_text, ground_truth),
        "helpfulness": judge_helpfulness(query, response_text),
    }
    for name, score in scores.items():
        llm_call_span.set_attribute(f"gen_ai.evaluation.{name}", score)
        llm_call_span.add_event(
            f"judge.{name}",
            attributes={
                "gen_ai.evaluation.name": name,
                "gen_ai.evaluation.score": score,
                "gen_ai.evaluation.judge_model": "gpt-4o",
            },
        )
    if scores["hallucination"] < 0.3:
        bad_case_queue.put(
            {
                "trace_id": format(llm_call_span.get_span_context().trace_id, "032x"),
                "query": query,
                "response": response_text,
                "scores": scores,
            }
        )
    return scores


if __name__ == "__main__":
    # 演示：让 1% 采样能命中（这里循环足够多次）
    random.seed(0)
    sampled = 0
    for i in range(500):
        with tracer.start_as_current_span(f"chat.gpt-4o.{i}") as span:
            response = f"answer {i}"
            scores = with_judge(span, response, f"q{i}")
            if scores is not None:
                sampled += 1
    print(f"sampled_judges: {sampled}, bad_cases_in_queue: {bad_case_queue.qsize()}")
    spans = exporter.get_finished_spans()
    judge_event_count = sum(1 for s in spans for e in s.events if e.name.startswith("judge."))
    print(f"judge events on spans: {judge_event_count}")
