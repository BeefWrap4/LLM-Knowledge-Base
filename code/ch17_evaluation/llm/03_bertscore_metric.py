# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.2.3 BERTScore
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: bert-score, torch, transformers
# run: python 03_bertscore_metric.py
# expected_runtime: 30-60s (first run downloads model)
# expected_output: BERTScore Precision/Recall/F1 around 0.85
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - How does BERTScore capture semantic similarity that BLEU misses?
# - What is the impact of choosing different backbone models for BERTScore?
# - When can BERTScore give misleading results?

"""BERTScore 计算示例。

BERTScore 使用预训练语言模型的上下文嵌入计算生成文本和参考文本之间的语义相似度。
"""


def compute_bertscore_demo() -> None:
    try:
        from bert_score import score
    except ImportError:
        print("[mock] bert_score 未安装。模拟 BERTScore 输出。")
        print("BERTScore Precision: 0.8500")
        print("BERTScore Recall:    0.8700")
        print("BERTScore F1:        0.8600")
        return

    references = ["The cat is sitting on the mat."]
    candidates = ["A feline rests upon the rug."]

    try:
        P, R, F1 = score(
            candidates,
            references,
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            verbose=True,
        )

        print(f"BERTScore Precision: {P.mean().item():.4f}")
        print(f"BERTScore Recall:    {R.mean().item():.4f}")
        print(f"BERTScore F1:        {F1.mean().item():.4f}")
        # 典型输出（即使词语完全不同，语义相似度高）：
        # BERTScore Precision: ~0.8500
        # BERTScore Recall:    ~0.8700
        # BERTScore F1:        ~0.8600
    except Exception as exc:
        print(f"[mock] 模型加载失败 ({exc})。模拟输出。")
        print("BERTScore Precision: 0.8500")
        print("BERTScore Recall:    0.8700")
        print("BERTScore F1:        0.8600")


if __name__ == "__main__":
    compute_bertscore_demo()
