# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.2 Ragas 框架
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: ragas>=0.4, openai, sentence-transformers
# run: python 06_ragas_evaluation.py
# expected_runtime: <2s (mock mode) / workload-dependent (real)
# expected_output: Offline configuration or measured per-sample metrics followed by OK
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Explain the difference between Faithfulness and Answer Relevancy.
# - Why does Context Recall require a reference answer or reference contexts?
# - Which Ragas v0.4 APIs replaced the legacy evaluate()/wrapper workflow?

"""Ragas v0.4 collections API 示例。

默认 ``LLM_MOCK=1``，不导入评估依赖、不读取密钥、不加载模型且不联网。
``LLM_MOCK=0`` 才会用 ``OPENAI_MODEL``（默认 ``gpt-5.6``）和本地 embedding
运行真实评估。真实模式失败会报错，不会退化为看似成功的模拟分数。
"""

import asyncio
import os
from pathlib import Path as _Path_setup
from typing import Any

_code_root = _Path_setup(__file__).resolve().parent.parent.parent


EVAL_ROWS = [
    {
        "user_input": "什么是 Transformer 的注意力机制？",
        "response": ("注意力机制通过 Query、Key、Value 计算 token 间关系，使模型能动态聚合不同位置的信息。"),
        "retrieved_contexts": [
            "注意力机制通过 Q、K、V 三个矩阵计算 token 间的关系。",
            "自注意力可以捕捉序列中任意两个位置之间的依赖关系。",
        ],
        "reference": "注意力机制通过 Q、K、V 计算，让模型聚合序列不同位置的信息。",
    },
    {
        "user_input": "CPython 中的 GIL 是什么？",
        "response": ("GIL 是 CPython 的全局解释器锁，通常限制同一进程内多个线程同时执行 Python 字节码。"),
        "retrieved_contexts": [
            "GIL 是 CPython 解释器中的全局锁。",
            "CPU 密集型多线程代码通常不能借此实现 Python 字节码的多核并行。",
        ],
        "reference": "GIL 是 CPython 的全局解释器锁，会限制 Python 字节码的线程级并行。",
    },
]


async def _run_real_evaluation() -> list[dict[str, float]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("真实模式需要 OPENAI_API_KEY；默认请使用 LLM_MOCK=1")

    embedding_path = _Path_setup(
        os.environ.get(
            "RAGAS_EMBEDDING_MODEL",
            str(_code_root / "models" / "bge-small-zh-v1.5"),
        )
    )
    if not (embedding_path / "config.json").is_file():
        raise RuntimeError(
            "真实模式需要本地 embedding 模型目录及 config.json；可用 RAGAS_EMBEDDING_MODEL 指定路径"
        )

    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError("真实模式需要 ragas>=0.4、openai 和 sentence-transformers") from exc

    client = AsyncOpenAI()
    try:
        llm = llm_factory(
            os.environ.get("OPENAI_MODEL", "gpt-5.6"),
            client=client,
        )
        embeddings = HuggingFaceEmbeddings(model=str(embedding_path))

        faithfulness = Faithfulness(llm=llm)
        answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        context_recall = ContextRecall(llm=llm)
        context_precision = ContextPrecision(llm=llm)

        rows: list[dict[str, float]] = []
        for item in EVAL_ROWS:
            results = await asyncio.gather(
                faithfulness.ascore(
                    user_input=item["user_input"],
                    response=item["response"],
                    retrieved_contexts=item["retrieved_contexts"],
                ),
                answer_relevancy.ascore(
                    user_input=item["user_input"],
                    response=item["response"],
                ),
                context_recall.ascore(
                    user_input=item["user_input"],
                    reference=item["reference"],
                    retrieved_contexts=item["retrieved_contexts"],
                ),
                context_precision.ascore(
                    user_input=item["user_input"],
                    reference=item["reference"],
                    retrieved_contexts=item["retrieved_contexts"],
                ),
            )
            rows.append(
                {
                    "faithfulness": float(results[0].value),
                    "answer_relevancy": float(results[1].value),
                    "context_recall": float(results[2].value),
                    "context_precision": float(results[3].value),
                }
            )
        return rows
    finally:
        await client.close()


def run_ragas_evaluation() -> list[dict[str, Any]]:
    """运行示意或真实评估，并返回逐样本结果。"""
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[mock] Ragas v0.4 collections 配置；未加载模型、未调用 API、未生成分数")
        print("  metrics: Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision")
        return [{"sample_id": index, "status": "not_measured"} for index, _ in enumerate(EVAL_ROWS, start=1)]

    rows = asyncio.run(_run_real_evaluation())
    for index, scores in enumerate(rows, start=1):
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in scores.items())
        print(f"sample_{index}: {rendered}")
    return rows


if __name__ == "__main__":
    run_ragas_evaluation()
    print("OK")
