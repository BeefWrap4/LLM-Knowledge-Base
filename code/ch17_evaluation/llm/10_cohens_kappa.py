# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.8.2 评分者间一致性
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy, scikit-learn, statsmodels
# run: python 10_cohens_kappa.py
# expected_runtime: <2s
# expected_output: Cohen's Kappa and Fleiss' Kappa values with interpretation
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - How does Cohen's Kappa correct for chance agreement?
# - When should you use Fleiss' Kappa instead of Cohen's Kappa?
# - Why is there no universal "acceptable" Kappa threshold?

"""Cohen's Kappa 计算示例。

当多个评估者对同一批回答评分时，需要衡量他们的评分一致性。
Likert 等有序标签应同时考虑加权 Kappa。阈值标签只是历史经验描述，
不能代替原始一致率、类别分布、置信区间和业务后果。
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def landis_koch_label(kappa: float) -> str:
    """返回常见的 Landis-Koch 经验标签；不代表通用验收阈值。"""
    if kappa < 0:
        return "一致性差于随机（Poor）"
    elif kappa < 0.20:
        return "轻微一致（Slight）"
    elif kappa < 0.40:
        return "一般一致（Fair）"
    elif kappa < 0.60:
        return "中等一致（Moderate）"
    elif kappa < 0.80:
        return "高度一致（Substantial）"
    else:
        return "几乎完全一致（Almost Perfect）"


def main() -> None:
    # 两个评估者对 10 个回答的 Likert 评分 (1-5)
    rater_a = [4, 5, 3, 4, 5, 2, 4, 3, 5, 4]
    rater_b = [4, 5, 3, 5, 4, 2, 4, 4, 5, 4]

    kappa = cohen_kappa_score(rater_a, rater_b)
    weighted_kappa = cohen_kappa_score(rater_a, rater_b, weights="quadratic")
    raw_agreement = np.mean(np.asarray(rater_a) == np.asarray(rater_b))
    print(f"Raw agreement: {raw_agreement:.4f}")
    print(f"Cohen's Kappa (unweighted): {kappa:.4f}")
    print(f"Cohen's Kappa (quadratic):  {weighted_kappa:.4f}")
    print(f"经验标签（非验收线）: {landis_koch_label(kappa)}")

    # Fleiss' Kappa（多个评估者）
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa

        # 3 位评估者对 5 个回答的评分分布
        # 每行是一个回答，每列是某个评分被选中的次数
        table = np.array(
            [
                [0, 0, 0, 2, 1],  # 回答1: 2人选4分，1人选5分
                [0, 0, 0, 1, 2],  # 回答2: 1人选4分，2人选5分
                [0, 0, 2, 1, 0],  # 回答3: 2人选3分，1人选4分
                [0, 1, 1, 1, 0],  # 回答4: 1人选2分，1人选3分，1人选4分
                [0, 0, 0, 2, 1],  # 回答5: 2人选4分，1人选5分
            ]
        )
        fkappa = fleiss_kappa(table)
        print(f"Fleiss' Kappa: {fkappa:.4f}")
        print(f"经验标签（非验收线）: {landis_koch_label(fkappa)}")
    except ImportError:
        print("[SKIP] statsmodels 未安装，跳过 Fleiss' Kappa")


if __name__ == "__main__":
    main()
    print("OK")
