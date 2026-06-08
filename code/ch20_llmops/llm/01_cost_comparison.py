# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.1.2 LLM 应用生命周期的特殊性
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 01_cost_comparison.py
# expected_runtime: < 1s
# expected_output: Two cost dicts comparing traditional ML and LLM inference cost models
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2012-llm-应用生命周期的特殊性
# Interview hooks:
#  - 为什么 LLM 推理成本是不可预测的？
#  - 传统 ML 与 LLM 推理在成本模型上的本质差异是什么？
#  - 为什么 Token 计量成为 LLMOps 核心指标？


class CostComparison:
    """传统 ML vs LLM 的成本特征差异"""

    @staticmethod
    def traditional_ml_cost(predictions_per_day: int, gpu_cost_per_hour: float = 3.0):
        """传统 ML：固定 GPU 实例成本"""
        daily_cost = 24 * gpu_cost_per_hour  # GPU 24小时运行
        cost_per_1k = daily_cost / (predictions_per_day / 1000)
        return {
            "daily_cost": daily_cost,
            "cost_per_1k_predictions": cost_per_1k,
            "cost_variance": "固定（无波动）",
        }

    @staticmethod
    def llm_cost(prompt_tokens_per_day: int, completion_tokens_per_day: int, model: str = "gpt-4o"):
        """LLM：按 Token 计费，成本波动大"""
        # 2026年参考价格（每百万 token）
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4": {"input": 3.00, "output": 15.00},
            "claude-haiku-4": {"input": 0.25, "output": 1.25},
        }
        p = pricing.get(model, pricing["gpt-4o-mini"])
        daily_cost = (
            prompt_tokens_per_day / 1_000_000 * p["input"]
            + completion_tokens_per_day / 1_000_000 * p["output"]
        )
        return {
            "daily_cost": daily_cost,
            "cost_per_1k_predictions": daily_cost / (prompt_tokens_per_day / 1000),
            "cost_variance": "按 Token 波动（Prompt 长度变化影响大）",
        }


if __name__ == "__main__":
    trad = CostComparison.traditional_ml_cost(predictions_per_day=100_000)
    llm = CostComparison.llm_cost(
        prompt_tokens_per_day=50_000_000,
        completion_tokens_per_day=20_000_000,
        model="gpt-4o",
    )
    print("=== Traditional ML cost ===")
    print(trad)
    print("=== LLM cost (gpt-4o) ===")
    print(llm)
