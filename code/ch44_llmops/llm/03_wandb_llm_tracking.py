# ---
# chapter: 44
# topic: LLMOps 生命周期与持续交付
# topic_id: llmops.wandb_llm_tracking
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: wandb, openai (live requires LLM_MOCK=0 and LLM_REAL_API=1)
# run: python 03_wandb_llm_tracking.py
# expected_runtime: < 1s (mocked) / depends on API (live)
# expected_output: W&B run logged with results table; accuracy reported
# ---
# See: ../../../44_LLMOps生命周期与持续交付.md
# Interview hooks:
#  - MLflow 与 W&B 在 LLM 实验追踪场景下的主要区别是什么？
#  - 为什么 W&B 的 Table 对 LLM 样本级调试更有优势？
#  - 在没有 W&B 账号时如何离线模拟其核心数据流？

import os
import time

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

def main():
    live_api = os.environ.get("LLM_REAL_API") == "1" and os.environ.get("LLM_MOCK") == "0"
    if not live_api:
        return _offline_mock()
    if wandb is None or OpenAI is None:
        raise RuntimeError("LLM_REAL_API=1 requires both wandb and openai packages")

    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")

    # 初始化 W&B
    wandb.init(
        project="llm-qa-evaluation",
        name=f"experiment-{wandb.util.generate_id()}",
        config={
            "model": model,
            "temperature": 0.1,
            "max_tokens": 200,
            "prompt_version": "v3_expert",
            "retrieval_top_k": 5,
        },
    )

    client = OpenAI()

    # 创建 W&B Table 记录每个样本的详细结果
    results_table = wandb.Table(columns=["query", "expected", "predicted", "correct", "latency_ms", "tokens"])

    test_data = [
        ("What is Python?", "A programming language"),
        ("Explain recursion", "A function that calls itself"),
    ]

    for query, expected in test_data:
        started_at = time.perf_counter()
        model_kwargs = (
            {
                "reasoning_effort": "none",
                "max_completion_tokens": wandb.config.max_tokens,
            }
            if model.startswith("gpt-5.6")
            else {"max_tokens": wandb.config.max_tokens}
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=wandb.config.temperature,
            **model_kwargs,
        )
        predicted = response.choices[0].message.content
        latency_ms = (time.perf_counter() - started_at) * 1000
        is_correct = expected.lower() in predicted.lower()

        results_table.add_data(
            query,
            expected,
            predicted[:200],
            is_correct,
            latency_ms,
            response.usage.total_tokens,
        )

    # 记录到 W&B
    wandb.log(
        {
            "accuracy": sum(1 for r in results_table.data if r[3]) / len(results_table.data),
            "results_table": results_table,
        }
    )

    wandb.finish()
    print("OK")


def _offline_mock():
    """离线 mock：模拟 W&B Table 与 accuracy 计算。"""
    rows = [
        ("What is Python?", "A programming language", "Answer for: What is Python?", False, 50, 80),
        (
            "Explain recursion",
            "A function that calls itself",
            "Answer for: Explain recursion",
            False,
            50,
            80,
        ),
    ]
    accuracy = sum(1 for r in rows if r[3]) / len(rows)
    print(f"[offline] accuracy={accuracy:.2%}, table_rows={len(rows)}")
    print("OK")


if __name__ == "__main__":
    main()
