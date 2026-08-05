# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.token_estimator
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: tiktoken (fallback estimator if missing)
# run: python 11_token_estimator.py
# expected_runtime: < 1s
# expected_output: Token count and cost estimate based on an injected illustrative rate card
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - 为什么应优先使用模型返回的 usage，而不是把本地估算当账单？
#  - 模型名无法被本地 tokenizer 识别时如何安全降级？
#  - 价格与上下文窗口为何必须按供应商当前文档注入？

import os
from dataclasses import dataclass

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None  # type: ignore
    _HAS_TIKTOKEN = False


@dataclass(frozen=True)
class ModelCostConfig:
    input_usd_per_million: float
    output_usd_per_million: float
    context_window_tokens: int
    source: str


class TokenEstimator:
    """规划阶段估算器；最终计费应以供应商 usage 与账单为准。"""

    @classmethod
    def count_tokens(cls, text: str, model: str) -> int:
        if _HAS_TIKTOKEN:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except (KeyError, ValueError):
                try:
                    encoding = tiktoken.get_encoding("o200k_base")
                except Exception:
                    encoding = None
            if encoding is not None:
                try:
                    return len(encoding.encode(text))
                except Exception:
                    pass
        return cls._estimate_tokens(text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗估：中文按 1.5 字符/Token、其他按 4 字符/Token；只用于容量规划。"""
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return max(1, int(chinese_chars / 1.5 + other_chars / 4))

    @classmethod
    def estimate_cost(
        cls,
        prompt: str,
        expected_output_tokens: int,
        model: str,
        config: ModelCostConfig,
    ) -> dict[str, float | int | str]:
        if expected_output_tokens < 0 or config.context_window_tokens <= 0:
            raise ValueError("output tokens must be non-negative and context window must be positive")
        input_tokens = cls.count_tokens(prompt, model)
        input_cost = input_tokens / 1_000_000 * config.input_usd_per_million
        output_cost = expected_output_tokens / 1_000_000 * config.output_usd_per_million
        return {
            "model": model,
            "input_tokens_estimated": input_tokens,
            "output_tokens_estimated": expected_output_tokens,
            "input_cost_usd_estimated": round(input_cost, 6),
            "output_cost_usd_estimated": round(output_cost, 6),
            "total_cost_usd_estimated": round(input_cost + output_cost, 6),
            "context_window_used_pct_estimated": round(
                input_tokens / config.context_window_tokens * 100,
                2,
            ),
            "rate_source": config.source,
        }


if __name__ == "__main__":
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")
    # 默认值是教学输入，不代表供应商当前价格或上下文上限。
    demo_config = ModelCostConfig(
        input_usd_per_million=float(os.environ.get("LLM_INPUT_USD_PER_MILLION", "1")),
        output_usd_per_million=float(os.environ.get("LLM_OUTPUT_USD_PER_MILLION", "4")),
        context_window_tokens=int(os.environ.get("LLM_CONTEXT_WINDOW_TOKENS", "100000")),
        source=os.environ.get("LLM_RATE_SOURCE", "illustrative-demo-rate-card"),
    )
    prompt = "请解释 Python 中的 asyncio、协程和事件循环。" * 5
    estimate = TokenEstimator.estimate_cost(
        prompt,
        expected_output_tokens=200,
        model=model,
        config=demo_config,
    )
    print(estimate)
    print("OK")
