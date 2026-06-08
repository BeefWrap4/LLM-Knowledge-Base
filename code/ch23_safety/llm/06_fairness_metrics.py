# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.4.4 公平性指标
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 06_fairness_metrics.py
# expected_runtime: <1s
# expected_output: 公平性指标计算结果 + "OK"
# ---
# See: ../tutorial/23_AI安全与伦理.md#2344-公平性指标
# Interview hooks:
#   1. Demographic Parity、Equalized Odds、Disparate Impact三个指标有什么区别？
#   2. 美国法律的"80%规则"具体含义是什么？
#   3. 在贷款审批模型中如何应用这些公平性指标？
"""
公平性评估指标实现
面试中常被问到：如何量化AI模型的公平性？
"""


def demographic_parity(positive_rates: dict[str, float]) -> float:
    """
    人口统计均等（Demographic Parity）

    不同群体获得正面预测的比例应当相等。

    Args:
        positive_rates: {"群体A": 0.7, "群体B": 0.3}

    Returns:
        最大差异值（越小越公平）
    """
    values = list(positive_rates.values())
    return max(values) - min(values)


def equalized_odds(
    tpr: dict[str, float],  # True Positive Rate per group
    fpr: dict[str, float],  # False Positive Rate per group
) -> dict[str, float]:
    """
    机会均等（Equalized Odds）

    不同群体的TPR和FPR应该相等。
    比Demographic Parity更严格，因为它考虑了真实标签。
    """
    return {
        "tpr_disparity": max(tpr.values()) - min(tpr.values()),
        "fpr_disparity": max(fpr.values()) - min(fpr.values()),
        "fair": (
            max(tpr.values()) - min(tpr.values()) < 0.05 and max(fpr.values()) - min(fpr.values()) < 0.05
        ),
    }


def disparate_impact_ratio(positive_rate_privileged: float, positive_rate_unprivileged: float) -> float:
    """
    差异影响比率（Disparate Impact）

    美国法律常用的"80%规则"：如果比值 < 0.8，
    则存在差异影响（潜在歧视）。

    DI = P(positive|unprivileged) / P(positive|privileged)
    """
    if positive_rate_privileged == 0:
        return float("inf")
    return positive_rate_unprivileged / positive_rate_privileged


# ========== 公平性计算示例 ==========
if __name__ == "__main__":
    # 场景：贷款审批模型
    groups = {"男性申请人": 0.85, "女性申请人": 0.62}
    dp = demographic_parity(groups)
    di = disparate_impact_ratio(0.85, 0.62)

    print("=== 公平性指标计算（贷款审批场景）===")
    print(f"男性通过率: {groups['男性申请人']:.2%}")
    print(f"女性通过率: {groups['女性申请人']:.2%}")
    print(f"人口统计均等差异: {dp:.3f}")
    print(f"差异影响比率: {di:.3f}")
    print(f"合规判断: {'✅ 通过80%规则' if di >= 0.8 else '❌ 违反80%规则，需审查'}")

    # Equalized Odds示例
    tpr = {"男性": 0.80, "女性": 0.65}
    fpr = {"男性": 0.10, "女性": 0.15}
    eo = equalized_odds(tpr, fpr)
    print("\n机会均等（Equalized Odds）:")
    print(f"  TPR差异: {eo['tpr_disparity']:.3f}")
    print(f"  FPR差异: {eo['fpr_disparity']:.3f}")
    print(f"  公平判定: {'✅ 公平' if eo['fair'] else '❌ 不公平'}")
