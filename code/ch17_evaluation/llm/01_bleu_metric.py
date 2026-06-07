# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.2.1 BLEU
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: nltk, sacrebleu
# run: python 01_bleu_metric.py
# expected_runtime: <5s
# expected_output: BLEU-1, BLEU-4 scores and SacreBLEU score
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Why is BLEU unsuitable for evaluating chatbots?
# - How does BLEU's brevity penalty work?
# - When should you choose SacreBLEU over NLTK BLEU?

"""BLEU 计算示例（使用 NLTK 和 SacreBLEU）。

BLEU 是机器翻译评估的经典指标，通过 n-gram 表面匹配衡量生成文本质量。
"""


def compute_bleu_demo() -> None:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("[mock] nltk 未安装。模拟 BLEU 输出。")
        print("BLEU-1: 0.8000")
        print("BLEU-4: 0.0000")
        print("SacreBLEU: 23.45")
        return

    # 参考文本和候选文本
    reference = [["The cat is on the mat".split()]]
    candidate = "The cat sits on the mat".split()

    # NLTK BLEU（使用平滑处理避免零值）
    smooth = SmoothingFunction().method1
    bleu_1 = sentence_bleu(
        reference[0],
        candidate,
        weights=(1, 0, 0, 0),
        smoothing_function=smooth,
    )
    bleu_4 = sentence_bleu(
        reference[0],
        candidate,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )

    print(f"BLEU-1: {bleu_1:.4f}")
    print(f"BLEU-4: {bleu_4:.4f}")
    # 输出：BLEU-1: 0.8000, BLEU-4: ~0.0000（n>2 时无匹配）

    try:
        from sacrebleu import corpus_bleu

        # SacreBLEU（推荐用于科研，结果可复现）
        refs = [["The cat is on the mat"]]
        hyps = ["The cat sits on the mat"]
        score = corpus_bleu(hyps, refs)
        print(f"SacreBLEU: {score.score:.2f}")
    except ImportError:
        print("[mock] sacrebleu 未安装，跳过 SacreBLEU。")


if __name__ == "__main__":
    compute_bleu_demo()
