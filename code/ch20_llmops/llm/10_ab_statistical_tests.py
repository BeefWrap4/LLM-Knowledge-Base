# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.4.3 统计显著性检验
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy, scipy
# run: python 10_ab_statistical_tests.py
# expected_runtime: < 1s
# expected_output: z-test / t-test / sample-size calculation results printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2043-统计显著性检验-⭐⭐
# Interview hooks:
#  - 双比例 Z 检验与 Welch T 检验分别适用于什么类型指标？
#  - 怎么估算"要检测 5% 提升需要多少样本"？
#  - p 值与置信区间在 A/B 测试报告里该如何正确呈现？

import numpy as np
from scipy import stats


class ABStatisticalTests:
    """A/B 测试常用统计检验"""

    @staticmethod
    def two_proportion_z_test(
        successes_a: int, n_a: int,
        successes_b: int, n_b: int,
        alpha: float = 0.05,
    ) -> dict:
        """双比例 Z 检验（用于二分类指标）"""
        p_a = successes_a / n_a
        p_b = successes_b / n_b
        p_pool = (successes_a + successes_b) / (n_a + n_b)

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        z_score = (p_b - p_a) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        diff = p_b - p_a
        ci_margin = stats.norm.ppf(1 - alpha / 2) * np.sqrt(
            p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b
        )
        ci_lower = diff - ci_margin
        ci_upper = diff + ci_margin

        return {
            "test": "Two-Proportion Z-Test",
            "control_rate": p_a,
            "treatment_rate": p_b,
            "difference": diff,
            "relative_change_pct": (diff / p_a * 100) if p_a > 0 else float('inf'),
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value < alpha,
            "ci_95": (ci_lower, ci_upper),
        }

    @staticmethod
    def welch_t_test(
        values_a: list, values_b: list,
        alpha: float = 0.05,
    ) -> dict:
        """Welch's T 检验（用于连续指标：延迟/Token 数/评分）"""
        t_stat, p_value = stats.ttest_ind(values_b, values_a, equal_var=False)
        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        return {
            "test": "Welch's T-Test",
            "control_mean": mean_a,
            "treatment_mean": mean_b,
            "difference": mean_b - mean_a,
            "relative_change_pct": (mean_b - mean_a) / mean_a * 100 if mean_a else float('inf'),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
        }

    @staticmethod
    def compute_required_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """计算 A/B 测试所需的最小样本量"""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        h = 2 * np.arcsin(np.sqrt(baseline_rate + minimum_detectable_effect)) - \
            2 * np.arcsin(np.sqrt(baseline_rate))
        if h == 0:
            return 0
        n = 2 * ((z_alpha + z_beta) / h) ** 2
        return int(np.ceil(n))


if __name__ == "__main__":
    tester = ABStatisticalTests()

    # 示例1：正确率对比
    result1 = tester.two_proportion_z_test(
        successes_a=170, n_a=200,
        successes_b=188, n_b=200,
    )
    print(f"正确率对比: p={result1['p_value']:.4f}, 显著={result1['significant']}")

    # 示例2：连续指标（延迟）
    lat_a = np.random.normal(800, 100, size=200).tolist()
    lat_b = np.random.normal(820, 110, size=200).tolist()
    result2 = tester.welch_t_test(lat_a, lat_b)
    print(f"延迟对比: p={result2['p_value']:.4f}, 显著={result2['significant']}")

    # 示例3：所需样本量
    n_required = tester.compute_required_sample_size(
        baseline_rate=0.85,
        minimum_detectable_effect=0.05,
    )
    print(f"每组需要 {n_required} 个样本")
    print("OK")
