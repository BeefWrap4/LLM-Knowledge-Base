# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.1.2 LLM 应用生命周期的特殊性
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 01_cost_comparison.py
# expected_runtime: < 1s
# expected_output: Two cost dicts using an injected, explicitly illustrative LLM rate card
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2012-llm-应用生命周期的特殊性
# Interview hooks:
#  - 为什么 LLM 推理成本是不可预测的？
#  - 传统 ML 与 LLM 推理在成本模型上的本质差异是什么？
#  - 为什么价格必须从供应商账单/配置注入，而不能写死在代码里？

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRates:
    """每百万 Token 的输入/输出价格；由调用方按供应商当前账单配置注入。"""

    input_usd_per_million: float
    output_usd_per_million: float
    source: str


class CostComparison:
    """传统 ML vs LLM 的成本特征差异。"""

    @staticmethod
    def traditional_ml_cost(predictions_per_day: int, gpu_cost_per_hour: float = 3.0):
        """传统 ML：用固定实例小时费率演示容量成本。"""
        if predictions_per_day <= 0:
            raise ValueError("predictions_per_day must be positive")
        daily_cost = 24 * gpu_cost_per_hour
        return {
            "daily_cost": daily_cost,
            "cost_per_1k_predictions": daily_cost / (predictions_per_day / 1000),
            "cost_variance": "容量固定时较稳定；实际仍受扩缩容与利用率影响",
        }

    @staticmethod
    def llm_cost(
        prompt_tokens_per_day: int,
        completion_tokens_per_day: int,
        model: str,
        rates: TokenRates,
    ):
        """LLM：按 Token 计费；价格由外部 Rate Card 注入。"""
        if prompt_tokens_per_day <= 0 or completion_tokens_per_day < 0:
            raise ValueError("token counts must be non-negative and input must be positive")
        daily_cost = (
            prompt_tokens_per_day / 1_000_000 * rates.input_usd_per_million
            + completion_tokens_per_day / 1_000_000 * rates.output_usd_per_million
        )
        return {
            "model": model,
            "daily_cost": daily_cost,
            # 分母只取输入 Token，因此明确标为 blended，避免被误读成输入单价。
            "blended_cost_per_1k_input_tokens": daily_cost / (prompt_tokens_per_day / 1000),
            "cost_variance": "按输入/输出 Token 与供应商计费规则波动",
            "rate_source": rates.source,
        }


if __name__ == "__main__":
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
    # 1/4 只是可复现的教学费率，不代表任何供应商当前价格；生产中从账单配置注入。
    demo_rates = TokenRates(
        input_usd_per_million=float(os.environ.get("LLM_INPUT_USD_PER_MILLION", "1")),
        output_usd_per_million=float(os.environ.get("LLM_OUTPUT_USD_PER_MILLION", "4")),
        source=os.environ.get("LLM_RATE_SOURCE", "illustrative-demo-rate-card"),
    )
    traditional = CostComparison.traditional_ml_cost(predictions_per_day=100_000)
    llm = CostComparison.llm_cost(
        prompt_tokens_per_day=50_000_000,
        completion_tokens_per_day=20_000_000,
        model=model,
        rates=demo_rates,
    )
    print("=== Traditional ML cost ===")
    print(traditional)
    print(f"=== LLM cost ({model}; illustrative rates) ===")
    print(llm)
    print("OK")
