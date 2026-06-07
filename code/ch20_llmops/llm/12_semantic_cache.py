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
#  - 为什么 Prompt Caching 节省 50-90% 而应用层缓存只能 30-60%？

import hashlib
import json
import time
from typing import Optional


class SemanticCache:
    """LLM 响应缓存（精确匹配版；生产中常用语义相似度匹配）"""

    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: dict = {}
        self.threshold = similarity_threshold

    def _compute_hash(self, prompt: str, model: str, **params) -> str:
        key_data = json.dumps({
            "prompt": prompt,
            "model": model,
            "params": {k: v for k, v in sorted(params.items())},
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, prompt: str, model: str, **params) -> Optional[str]:
        """精确匹配缓存"""
        key = self._compute_hash(prompt, model, **params)
        if key in self.cache:
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
        total_hits = sum(v["hits"] for v in self.cache.values())
        return {
            "cache_entries": total,
            "total_hits": total_hits,
            "hit_rate": total_hits / max(total_hits + total, 1),
        }

    def estimated_savings(self, avg_cost_per_call: float = 0.01) -> float:
        """估算节省成本"""
        stats = self.get_cache_stats()
        return stats["total_hits"] * avg_cost_per_call


if __name__ == "__main__":
    cache = SemanticCache()

    def cached_llm_call(prompt: str, model: str = "gpt-4o-mini", **params) -> str:
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
    print("estimated savings: $", cache.estimated_savings())
