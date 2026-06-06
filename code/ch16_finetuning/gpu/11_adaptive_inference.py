# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.8.3 自适应推理引擎 (Test-Time Compute)
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only; 真实环境需 httpx/openai SDK)
# run: python 11_adaptive_inference.py
# expected_runtime: <2s
# expected_output: 演示 fast/balanced/deep 三档配置与 Self-Consistency 投票逻辑
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.8.3
# Interview hooks:
#   1. Test-Time Compute 的核心范式：训练计算 + 推理计算 = 最终能力？
#   2. Self-Consistency 的投票机制如何实现？多数投票 vs 加权投票？
#   3. 成本-质量权衡：Fast 模式 vs Deep 模式的 token 消耗比？典型 5-10x？

"""
自适应推理引擎 —— 根据问题难度动态调整推理计算量
2026 年 Claude 4.6/4.7 和 GPT-5.5 都支持多档推理模式
"""

from __future__ import annotations
import asyncio
import random
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelOutput:
    text: str
    usage: "Usage" = field(default_factory=lambda: Usage(0, 0))
    latency_ms: float = 0.0


@dataclass
class Usage:
    total_tokens: int = 0


@dataclass
class InferenceResult:
    answer: str
    complexity: str
    tokens_used: int
    latency_ms: float
    candidates: Optional[list] = None


class MockModelClient:
    """Mock 模型客户端 - 模拟多档推理返回"""
    def __init__(self):
        self.answers = [
            "简短回答 A", "简短回答 B", "简短回答 C",
            "详细推理过程: ... 得出结论 X", "另一个推理: ... 结论 Y",
            "第三种思路: ... 结论 X",
        ]

    async def generate(self, prompt: str, config: dict) -> ModelOutput:
        # 模拟单次推理
        await asyncio.sleep(0.01)
        idx = hash(prompt) % len(self.answers)
        return ModelOutput(
            text=self.answers[idx],
            usage=Usage(total_tokens=config.get("max_tokens", 512)),
            latency_ms=random.uniform(50, 200),
        )

    async def generate_n(self, prompt: str, config: dict, n: int) -> list[ModelOutput]:
        # Self-Consistency: 多次采样
        return [await self.generate(prompt, config) for _ in range(n)]


class AdaptiveInferenceEngine:
    """自适应推理引擎"""

    def __init__(self, model_client):
        self.model = model_client

    async def generate(
        self,
        query: str,
        complexity: str = "auto",   # "fast" | "balanced" | "deep" | "auto"
        max_think_tokens: Optional[int] = None,
    ) -> InferenceResult:
        """自适应推理: 根据复杂度选择推理策略"""
        if complexity == "auto":
            complexity = self._estimate_complexity(query)

        configs = {
            "fast": {
                "cot": False, "temperature": 0.3, "max_tokens": 512,
                "self_consistency_n": 1,
            },
            "balanced": {
                "cot": True, "temperature": 0.5, "max_tokens": 1024,
                "self_consistency_n": 3,
            },
            "deep": {
                "cot": True, "temperature": 0.7, "max_tokens": 2048,
                "self_consistency_n": 5, "verification": True,
            },
        }
        config = configs[complexity]
        prompt = self._build_prompt(query, config)

        if config["self_consistency_n"] == 1:
            response = await self.model.generate(prompt, config)
            return InferenceResult(
                answer=response.text,
                complexity=complexity,
                tokens_used=response.usage.total_tokens,
                latency_ms=response.latency_ms,
            )
        # Self-Consistency: 多次采样 + 投票
        responses = await self.model.generate_n(
            prompt, config, n=config["self_consistency_n"],
        )
        best_answer = self._vote(responses)
        return InferenceResult(
            answer=best_answer,
            complexity=complexity,
            candidates=[r.text for r in responses],
            tokens_used=sum(r.usage.total_tokens for r in responses),
            latency_ms=max(r.latency_ms for r in responses),
        )

    @staticmethod
    def _build_prompt(query: str, config: dict) -> str:
        if config.get("cot"):
            return f"请一步步思考, 然后回答: {query}"
        return query

    @staticmethod
    def _vote(responses: list[ModelOutput]) -> str:
        """多数投票 - 选择出现频次最高的答案"""
        from collections import Counter
        texts = [r.text for r in responses]
        if not texts:
            return ""
        most_common, _ = Counter(texts).most_common(1)[0]
        return most_common

    @staticmethod
    def _estimate_complexity(query: str) -> str:
        """估算查询复杂度（关键词启发式）"""
        complex_indicators = [
            "证明", "推导", "分析", "比较", "为什么",
            "optimize", "prove", "derive", "compare",
        ]
        simple_indicators = [
            "什么是", "定义", "简介", "who is", "what is",
        ]
        q_lower = query.lower()
        complex_score = sum(1 for w in complex_indicators if w in q_lower)
        simple_score = sum(1 for w in simple_indicators if w in q_lower)
        if complex_score > simple_score:
            return "deep"
        if simple_score > 0:
            return "fast"
        return "balanced"


if __name__ == "__main__":
    engine = AdaptiveInferenceEngine(MockModelClient())

    async def demo():
        queries = [
            ("什么是 Python?", "fast"),
            ("如何优化推荐系统的召回率?", "balanced"),
            ("证明 n^3 - n 能被 6 整除", "deep"),
        ]
        for q, expected in queries:
            result = await engine.generate(q, complexity=expected)
            print(f"Q: {q[:30]:<32} | mode={result.complexity:<8} | "
                  f"tokens={result.tokens_used} | "
                  f"candidates={len(result.candidates) if result.candidates else 1}")
        # Auto 模式演示
        for q in ["你好", "证明费马大定理", "解释 Transformer"]:
            r = await engine.generate(q, complexity="auto")
            print(f"[AUTO] Q: {q:<25} -> mode={r.complexity}")

    asyncio.run(demo())
    print("OK")
