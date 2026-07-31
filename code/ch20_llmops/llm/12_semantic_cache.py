# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.5.3 成本优化策略
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 12_semantic_cache.py
# expected_runtime: < 1s
# expected_output: Cache hit/miss flow demonstrated, savings estimate printed
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2053-成本优化策略-⭐⭐⭐⭐
# Interview hooks:
#  - 精确哈希缓存与语义缓存（向量相似度）的取舍？
#  - 缓存命中率的"分子分母"如何定义才合理？
#  - 为什么缓存节省率必须用真实流量与当前计费规则测量？

import hashlib
import json
import os
import time


class ExactPromptCache:
    """教学用精确哈希缓存；本例不做向量语义匹配。"""

    def __init__(self):
        self.cache: dict = {}
        self.lookups = 0
        self.hits = 0

    def _compute_hash(self, prompt: str, model: str, **params) -> str:
        key_data = json.dumps(
            {
                "prompt": prompt,
                "model": model,
                "params": {k: v for k, v in sorted(params.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, prompt: str, model: str, **params) -> str | None:
        """精确匹配缓存"""
        self.lookups += 1
        key = self._compute_hash(prompt, model, **params)
        if key in self.cache:
            self.hits += 1
            self.cache[key]["hits"] += 1
            return self.cache[key]["response"]
        return None

    def set(self, prompt: str, model: str, response: str, **params):
        """存入缓存"""
        key = self._compute_hash(prompt, model, **params)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "hits": 0,
        }

    def get_cache_stats(self) -> dict:
        """缓存统计"""
        total = len(self.cache)
        return {
            "cache_entries": total,
            "lookups": self.lookups,
            "total_hits": self.hits,
            "total_misses": self.lookups - self.hits,
            "hit_rate": self.hits / self.lookups if self.lookups else 0.0,
        }

    def estimated_savings(self, avoided_cost_per_hit_usd: float) -> float:
        """按调用方注入的实际/估算单次成本计算；不内置供应商价格。"""
        if avoided_cost_per_hit_usd < 0:
            raise ValueError("avoided_cost_per_hit_usd must be non-negative")
        return self.hits * avoided_cost_per_hit_usd


if __name__ == "__main__":
    cache = ExactPromptCache()

    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")

    def cached_llm_call(prompt: str, **params) -> str:
        cached = cache.get(prompt, model, **params)
        if cached:
            print("⚡ Cache hit! 节省一次 API 调用")
            return cached
        response = f"Response for: {prompt[:50]}..."
        cache.set(prompt, model, response, **params)
        return response

    # 第一次：未命中
    cached_llm_call("解释 Python 装饰器", temperature=0.1)
    # 第二次：命中
    cached_llm_call("解释 Python 装饰器", temperature=0.1)
    # 第三次：参数不同，未命中
    cached_llm_call("解释 Python 装饰器", temperature=0.7)

    print("cache stats:", cache.get_cache_stats())
    # 该默认值只是演示输入；生产中应使用实际账单归因得到的 avoided cost。
    avoided_cost = float(os.environ.get("LLM_AVOIDED_COST_PER_HIT_USD", "0.01"))
    print("estimated savings (illustrative): $", cache.estimated_savings(avoided_cost))
    print("OK")
