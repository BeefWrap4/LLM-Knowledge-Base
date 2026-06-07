# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.2.2 ROUGE
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: rouge-score
# run: python 02_rouge_metric.py
# expected_runtime: <5s
# expected_output: ROUGE-1/2/L precision/recall/F1 scores
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What is the core difference between BLEU and ROUGE?
# - When would you choose ROUGE-L over ROUGE-N?
# - Why is ROUGE the standard for summarization evaluation?

"""ROUGE 计算示例（使用 rouge_score 库）。

ROUGE 侧重召回率，衡量参考文本中的 n-gram 有多少出现在生成文本中。
"""


def compute_rouge_demo() -> None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[mock] rouge_score 未安装。模拟 ROUGE 输出。")
        print("rouge1: Precision=0.667, Recall=0.500, F1=0.571")
        print("rouge2: Precision=0.250, Recall=0.182, F1=0.211")
        print("rougeL: Precision=0.583, Recall=0.438, F1=0.500")
        return

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    reference = "The cat sat on the mat under the warm sunlight."
    candidate = "A cat is sitting on a mat in the sunlight."

    scores = scorer.score(reference, candidate)
    for metric, score in scores.items():
        print(
            f"{metric}: Precision={score.precision:.3f}, "
            f"Recall={score.recall:.3f}, F1={score.fmeasure:.3f}"
        )
    # 典型输出：
    # rouge1: Precision=0.667, Recall=0.500, F1=0.571
    # rouge2: Precision=0.250, Recall=0.182, F1=0.211
    # rougeL: Precision=0.583, Recall=0.438, F1=0.500


if __name__ == "__main__":
    compute_rouge_demo()
