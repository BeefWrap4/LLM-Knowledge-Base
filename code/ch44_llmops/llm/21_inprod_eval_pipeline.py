# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.inprod_eval_pipeline
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: opentelemetry-api, opentelemetry-sdk
# run: python 21_inprod_eval_pipeline.py
# expected_runtime: < 1s
# expected_output: Span list with judge evaluation events and bad-case queue contents
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - in-prod eval 为什么要与离线回归和人工复核互补？成本/样本偏差如何权衡？
#  - 如何按预算、风险与标注能力选择线上 Judge 采样率？
#  - bad case 自动入训练集的反馈回路有哪些工程陷阱（标签噪声、时序）？

import os
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

tracer = trace.get_tracer("in-prod-eval")
bad_case_queue: "queue.Queue" = queue.Queue()


def judge_relevance(query, response):
    return random.uniform(0.6, 0.99)


def judge_hallucination(response, ground_truth):
    return random.uniform(0.4, 0.95)


def judge_helpfulness(query, response):
    return random.uniform(0.5, 0.95)


def with_judge(
    llm_call_span,
    response_text: str,
    query: str,
    ground_truth=None,
    *,
    sampling_probability: float,
    judge_model: str,
    bad_case_threshold: float,
):
    """按注入的采样率挂载离线 Judge 分数；比例和阈值需由预算/标注集校准。"""
    if not 0 <= sampling_probability <= 1:
        raise ValueError("sampling_probability must be in [0, 1]")
    if random.random() > sampling_probability:
        return None  # 采样外，跳过

    scores = {
        "relevance": judge_relevance(query, response_text),
        "hallucination": judge_hallucination(response_text, ground_truth),
        "helpfulness": judge_helpfulness(query, response_text),
    }
    for name, score in scores.items():
        llm_call_span.set_attribute(f"app.evaluation.{name}", score)
        llm_call_span.add_event(
            f"judge.{name}",
            attributes={
                "app.evaluation.name": name,
                "app.evaluation.score": score,
                "app.evaluation.judge_model": judge_model,
            },
        )
    if scores["hallucination"] > bad_case_threshold:
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
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
    judge_model = os.environ.get("LLM_JUDGE_MODEL", model)
    sampling_probability = float(os.environ.get("LLM_JUDGE_SAMPLE_RATIO", "0.01"))
    bad_case_threshold = float(os.environ.get("LLM_BAD_CASE_THRESHOLD", "0.7"))
    # 这里的默认采样率/阈值只是教学策略参数，不是行业基准。
    random.seed(0)
    sampled = 0
    for i in range(500):
        with tracer.start_as_current_span(f"chat {model}") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.provider.name", "openai")
            span.set_attribute("gen_ai.request.model", model)
            response = f"answer {i}"
            scores = with_judge(
                span,
                response,
                f"q{i}",
                sampling_probability=sampling_probability,
                judge_model=judge_model,
                bad_case_threshold=bad_case_threshold,
            )
            if scores is not None:
                sampled += 1
    print(f"sampled_judges: {sampled}, bad_cases_in_queue: {bad_case_queue.qsize()}")
    spans = exporter.get_finished_spans()
    judge_event_count = sum(1 for s in spans for e in s.events if e.name.startswith("judge."))
    print(f"judge events on spans: {judge_event_count}")
    print("OK")
