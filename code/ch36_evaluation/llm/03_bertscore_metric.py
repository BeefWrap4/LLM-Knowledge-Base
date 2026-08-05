# ---
# chapter: 36
# topic: 大模型评估基础
# topic_id: evaluation.bertscore_metric
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: bert-score, torch, transformers
# run: python 03_bertscore_metric.py
# expected_runtime: <2s (mock mode) / model-dependent (real)
# expected_output: Skip notice in mock mode or measured BERTScore values in real mode
# ---
# See: ../../../36_大模型评估基础.md
# Interview hooks:
# - How does BERTScore capture semantic similarity that BLEU misses?
# - What is the impact of choosing different backbone models for BERTScore?
# - When can BERTScore give misleading results?

"""BERTScore 计算示例。

默认 ``LLM_MOCK=1`` 不加载或下载模型。设置 ``LLM_MOCK=0`` 后才执行真实计算，
并用 ``BERTSCORE_MODEL`` 固定 backbone；依赖或模型错误不会伪装成模拟分数。
"""

import os


def compute_bertscore_demo() -> tuple[float, float, float]:
    if os.environ.get("LLM_MOCK", "1") != "0":
        raise RuntimeError("LLM_MOCK=1：未加载模型；真实计算请显式设置 LLM_MOCK=0")

    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("真实模式需要 bert-score、torch 和 transformers") from exc

    references = ["The cat is sitting on the mat."]
    candidates = ["A feline rests upon the rug."]
    precision, recall, f1 = score(
        candidates,
        references,
        model_type=os.environ.get(
            "BERTSCORE_MODEL",
            "microsoft/deberta-xlarge-mnli",
        ),
        lang="en",
        verbose=True,
    )
    values = (
        precision.mean().item(),
        recall.mean().item(),
        f1.mean().item(),
    )
    print(f"BERTScore Precision: {values[0]:.4f}")
    print(f"BERTScore Recall:    {values[1]:.4f}")
    print(f"BERTScore F1:        {values[2]:.4f}")
    return values


if __name__ == "__main__":
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[SKIP] LLM_MOCK=1：未加载模型，未生成伪 BERTScore")
    else:
        compute_bertscore_demo()
    print("OK")
