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
#   2. 美国 EEOC adverse impact 筛查中的"80%规则"是什么？为什么它不是合规结论？
#   3. 在招聘筛选模型中如何应用这些公平性指标？
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
    """计算差异影响比率（Disparate Impact Ratio）。

    美国 EEOC《Uniform Guidelines on Employee Selection Procedures》中的
    four-fifths rule 是 adverse impact 的实务筛查经验规则：比率低于 0.8
    通常触发进一步审查，但它不是自动违法或合规的判定线；还要结合样本量、
    统计与实际显著性、岗位相关性以及适用法域进行审查。

    DI = P(positive|unprivileged) / P(positive|privileged)
    """
    if positive_rate_privileged == 0:
        return float("inf")
    return positive_rate_unprivileged / positive_rate_privileged


# ========== 公平性计算示例 ==========
if __name__ == "__main__":
    # 场景：招聘筛选。EEOC 的 four-fifths rule 针对雇佣选择程序，
    # 不应直接当成信贷、教育等其他法域的合规判定。
    groups = {"最高通过率群体": 0.85, "待评估群体": 0.62}
    dp = demographic_parity(groups)
    di = disparate_impact_ratio(0.85, 0.62)

    print("=== 公平性指标计算（招聘筛选场景）===")
    print(f"最高群体通过率: {groups['最高通过率群体']:.2%}")
    print(f"待评估群体通过率: {groups['待评估群体']:.2%}")
    print(f"人口统计均等差异: {dp:.3f}")
    print(f"差异影响比率: {di:.3f}")
    screening = (
        "未低于0.8；仍不能据此证明公平或法律合规"
        if di >= 0.8
        else "低于0.8，触发 adverse impact 深入审查；不等同于自动违法"
    )
    print(f"EEOC 80%经验筛查: {screening}")

    # Equalized Odds示例
    tpr = {"男性": 0.80, "女性": 0.65}
    fpr = {"男性": 0.10, "女性": 0.15}
    eo = equalized_odds(tpr, fpr)
    print("\n机会均等（Equalized Odds）:")
    print(f"  TPR差异: {eo['tpr_disparity']:.3f}")
    print(f"  FPR差异: {eo['fpr_disparity']:.3f}")
    print(f"  示例阈值判定: {'差异低于5%' if eo['fair'] else '至少一项差异达到5%，需审查'}")
    print("OK")
