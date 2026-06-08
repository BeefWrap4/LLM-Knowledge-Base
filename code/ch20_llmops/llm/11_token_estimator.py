# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.5.2 Token 计数与预估
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: tiktoken (fallback estimator if missing)
# run: python 11_token_estimator.py
# expected_runtime: < 1s
# expected_output: Token counts, cost estimates, and per-model comparison
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2052-token-计数与预估-⭐⭐⭐
# Interview hooks:
#  - tiktoken 与模型编码之间如何映射？
#  - 在没有 tiktoken 时如何粗略估算 Token 数？
#  - 上下文窗口使用率为什么是 LLMOps 关注指标？


try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None  # type: ignore
    _HAS_TIKTOKEN = False


class TokenEstimator:
    """Token 计数与成本预估器"""

    MODEL_ENCODING_MAP: dict[str, str] = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3": "cl100k_base",
    }

    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00, "context_window": 128000},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "context_window": 128000},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00, "context_window": 200000},
        "claude-haiku-4": {"input": 0.25, "output": 1.25, "context_window": 200000},
    }

    @classmethod
    def count_tokens(cls, text: str, model: str = "gpt-4o") -> int:
        """统计文本的 Token 数量"""
        if _HAS_TIKTOKEN:
            encoding_name = cls.MODEL_ENCODING_MAP.get(model, "cl100k_base")
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                return len(encoding.encode(text))
            except Exception:
                pass
        return cls._estimate_tokens(text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Token 估算（不依赖 tiktoken）：中文 1.5 字符/token，英文 4 字符/token。"""
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    @classmethod
    def estimate_cost(
        cls,
        prompt: str,
        expected_output_length: int = 200,
        model: str = "gpt-4o",
    ) -> dict[str, float]:
        """预估单次调用成本"""
        input_tokens = cls.count_tokens(prompt, model)
        output_tokens = expected_output_length
        pricing = cls.PRICING.get(model, cls.PRICING["gpt-4o-mini"])

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens_estimated": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "context_window_used_pct": round(input_tokens / pricing["context_window"] * 100, 1),
        }

    @classmethod
    def compare_models(cls, prompt: str, expected_output: int = 200) -> list[dict]:
        """对比不同模型的成本"""
        results: list[dict] = []
        for model in cls.PRICING:
            results.append(cls.estimate_cost(prompt, expected_output, model))
        results.sort(key=lambda x: x["total_cost"])
        return results


if __name__ == "__main__":
    estimator = TokenEstimator()

    prompt = "请详细解释 Python 中的异步编程模型，包括 asyncio、协程和事件循环的概念。" * 5

    cost = estimator.estimate_cost(prompt, model="gpt-4o")
    print(f"GPT-4o: ${cost['total_cost']:.4f} ({cost['input_tokens']} input tokens)")

    comparison = estimator.compare_models(prompt)
    print("\n=== 模型成本对比 ===")
    for c in comparison:
        print(f"{c['model']}: ${c['total_cost']:.6f}")
