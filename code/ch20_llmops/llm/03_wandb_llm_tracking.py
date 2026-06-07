# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.2.3 Weights & Biases (W&B) 实战
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: wandb, openai (mocked fallback if unavailable)
# run: python 03_wandb_llm_tracking.py
# expected_runtime: < 1s (mocked) / depends on API (live)
# expected_output: W&B run logged with results table; accuracy reported
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2023-weights--biases-wb-实战-⭐⭐⭐
# Interview hooks:
#  - MLflow 与 W&B 在 LLM 实验追踪场景下的主要区别是什么？
#  - 为什么 W&B 的 Table 对 LLM 样本级调试更有优势？
#  - 在没有 W&B 账号时如何离线模拟其核心数据流？

import os

try:
    import wandb
    _HAS_WANDB = bool(os.getenv("WANDB_API_KEY"))
except ImportError:
    wandb = None  # type: ignore
    _HAS_WANDB = False

try:
    from openai import OpenAI
    _HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _mock_openai():
    class _Choice:
        def __init__(self, content, ct, tt):
            self.message = type("M", (), {"content": content})()
            self.usage = type("U", (), {"completion_tokens": ct, "total_tokens": tt})()

    class _Mock:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature=0.1, max_tokens=200):
                    user_msg = messages[-1]["content"]
                    return _Choice(f"Answer for: {user_msg}", ct=50, tt=80)
    return _Mock()


def main():
    if wandb is None or not _HAS_WANDB:
        print("wandb not available — running offline mock to demonstrate data flow")
        return _offline_mock()

    # 初始化 W&B
    wandb.init(
        project="llm-qa-evaluation",
        name=f"experiment-{wandb.util.generate_id()}",
        config={
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 200,
            "prompt_version": "v3_expert",
            "retrieval_top_k": 5,
        }
    )

    client = OpenAI() if _HAS_OPENAI and OpenAI is not None else _mock_openai()

    # 创建 W&B Table 记录每个样本的详细结果
    results_table = wandb.Table(
        columns=["query", "expected", "predicted", "correct", "latency_ms", "tokens"]
    )

    test_data = [
        ("What is Python?", "A programming language"),
        ("Explain recursion", "A function that calls itself"),
    ]

    for query, expected in test_data:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            temperature=wandb.config.temperature,
            max_tokens=wandb.config.max_tokens,
        )
        predicted = response.choices[0].message.content
        is_correct = expected.lower() in predicted.lower()

        results_table.add_data(
            query, expected, predicted[:200], is_correct,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )

    # 记录到 W&B
    wandb.log({
        "accuracy": sum(1 for r in results_table.data if r[3]) / len(results_table.data),
        "results_table": results_table,
    })

    wandb.finish()


def _offline_mock():
    """离线 mock：模拟 W&B Table 与 accuracy 计算。"""
    rows = [
        ("What is Python?", "A programming language", "Answer for: What is Python?", False, 50, 80),
        ("Explain recursion", "A function that calls itself", "Answer for: Explain recursion", False, 50, 80),
    ]
    accuracy = sum(1 for r in rows if r[3]) / len(rows)
    print(f"[offline] accuracy={accuracy:.2%}, table_rows={len(rows)}")


if __name__ == "__main__":
    main()
