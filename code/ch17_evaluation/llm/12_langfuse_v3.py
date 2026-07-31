# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.11.2 Langfuse Experiment Runner
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langfuse, openai
# run: python 12_langfuse_v3.py
# expected_runtime: <2s (mock mode)
# expected_output: Current Langfuse experiment flow followed by OK
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What does run_experiment automate, and what does it not validate for you?
# - When should an evaluator be deterministic code instead of LLM-as-Judge?
# - Why must a short-lived process call langfuse.flush()?

"""Langfuse 当前 Python SDK 的本地数据集实验示例。

示例使用 ``get_client``、``Evaluation``、``run_experiment`` 以及 Langfuse 的
OpenAI drop-in client。默认 ``LLM_MOCK=1``，不导入 SDK、不读取凭据、不联网。
"""

import os
from typing import Any


def run_langfuse_experiment_demo() -> Any:
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[mock] Langfuse 当前 Experiment Runner 流程（未创建 trace）")
        print("  get_client() -> run_experiment(data, task, evaluators)")
        print("  task -> langfuse.openai.OpenAI().responses.create(...)")
        print("  evaluator -> Evaluation(name, value, comment)")
        print("  short-lived process -> langfuse.flush()")
        return None

    required = [
        name
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OPENAI_API_KEY")
        if not os.environ.get(name)
    ]
    if required:
        raise RuntimeError(f"真实模式缺少环境变量: {', '.join(required)}")

    try:
        from langfuse import Evaluation, get_client
        from langfuse.openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("真实模式需要 langfuse 和 openai") from exc

    langfuse = get_client()
    openai_client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")

    def answer_task(*, item: dict[str, str], **_: Any) -> str:
        response = openai_client.responses.create(
            model=model,
            input=item["input"],
            reasoning={"effort": "none"},
        )
        return response.output_text

    def exact_match_evaluator(
        *,
        output: str,
        expected_output: str | None,
        **_: Any,
    ) -> Any:
        matched = bool(expected_output and expected_output.casefold().strip() in output.casefold().strip())
        return Evaluation(
            name="contains_expected_answer",
            value=1.0 if matched else 0.0,
            comment="Deterministic substring check; not an LLM judge.",
        )

    result = langfuse.run_experiment(
        name="ch17-geography-smoke",
        description="Current SDK local-dataset experiment example",
        data=[
            {"input": "法国的首都是哪里？只回答城市名。", "expected_output": "巴黎"},
            {"input": "德国的首都是哪里？只回答城市名。", "expected_output": "柏林"},
        ],
        task=answer_task,
        evaluators=[exact_match_evaluator],
        metadata={"model": model, "reasoning_effort": "none"},
        max_concurrency=2,
    )
    print(result.format())
    langfuse.flush()
    return result


if __name__ == "__main__":
    run_langfuse_experiment_demo()
    print("OK")
