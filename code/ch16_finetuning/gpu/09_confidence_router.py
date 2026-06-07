# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.7.3 基于置信度的动态路由
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only)
# run: python 09_confidence_router.py
# expected_runtime: <2s
# expected_output: 演示 Cascade 路由在 3 个 mock query 上的命中层级与置信度
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.7.3
# Interview hooks:
#   1. 端云协同的三种动态调度策略（Cascade / Prediction / Hybrid）各自优劣？
#   2. 模型置信度如何估计？token 平均概率 vs 序列级 log-likelihood 哪个更稳？
#   3. privacy_level=high 直接路由到端侧的工程意义？数据合规边界？

"""
基于置信度的动态调度路由 —— 端云协同 2026 面试热点
"""

from __future__ import annotations
import random
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    text: str
    token_probs: list = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class RouterResult:
    tier: str
    confidence: float
    response: str
    latency_ms: float


class ConfidenceRouter:
    """
    基于模型置信度的动态调度路由
    策略：端侧先尝试，置信度低于阈值则上云
    """

    def __init__(
        self,
        device_model,
        edge_model,
        cloud_model,
        device_threshold: float = 0.8,
        edge_threshold: float = 0.75,
    ):
        self.models = {
            "device": device_model,
            "edge": edge_model,
            "cloud": cloud_model,
        }
        self.thresholds = {
            "device": device_threshold,
            "edge": edge_threshold,
        }

    def route(self, query: str, context: Optional[dict] = None) -> RouterResult:
        """
        动态路由决策
        返回: {"tier": "device|edge|cloud", "confidence": float, "response": str}
        """
        # 策略1: 隐私检查 - 敏感数据直接端侧处理
        if context and context.get("privacy_level") == "high":
            return self._execute("device", query)

        # 策略2: 逐层尝试 (Cascade)
        for tier in ["device", "edge"]:
            result = self._execute(tier, query)
            if result.confidence >= self.thresholds[tier]:
                return result

        # 策略3: 云端兜底
        return self._execute("cloud", query)

    def _execute(self, tier: str, query: str) -> RouterResult:
        """在指定层级执行推理"""
        model = self.models[tier]
        response = model.generate(query)
        confidence = self._compute_confidence(response)
        return RouterResult(
            tier=tier,
            confidence=confidence,
            response=response.text,
            latency_ms=response.latency_ms,
        )

    @staticmethod
    def _compute_confidence(response) -> float:
        """计算模型输出的置信度（平均 token 概率）"""
        if hasattr(response, "token_probs") and response.token_probs:
            return sum(response.token_probs) / len(response.token_probs)
        return 0.5  # 默认中等置信度


# ========== Mock 模型（用于本地演示）==========
class MockModel:
    """模拟不同层级模型的输出行为"""

    def __init__(self, tier: str, base_conf: float, latency: float):
        self.tier = tier
        self.base_conf = base_conf
        self.latency = latency

    def generate(self, query: str) -> ModelResponse:
        # 模拟生成：短文本 + 与 base_conf 接近的置信度
        n_tokens = random.randint(8, 20)
        token_probs = [
            max(0.0, min(1.0, self.base_conf + random.uniform(-0.1, 0.1)))
            for _ in range(n_tokens)
        ]
        return ModelResponse(
            text=f"[{self.tier}] 模拟回答 for: {query[:30]}",
            token_probs=token_probs,
            latency_ms=self.latency + random.uniform(-5, 5),
        )


if __name__ == "__main__":
    # 端侧高置信 (简单问题), 边缘中等, 云端兜底
    device = MockModel("device", base_conf=0.85, latency=80)    # 3B-7B
    edge = MockModel("edge", base_conf=0.78, latency=300)       # 13B-34B
    cloud = MockModel("cloud", base_conf=0.92, latency=1500)    # 70B+

    router = ConfidenceRouter(
        device_model=device,
        edge_model=edge,
        cloud_model=cloud,
        device_threshold=0.8,
        edge_threshold=0.75,
    )

    test_queries = [
        ("你好", {}),                                  # 简单问候 - 应命中 device
        ("写一个 SQL 查询统计用户数", {}),                  # 中等 - 应命中 edge
        ("证明 n^3-n 能被 6 整除", {}),                   # 复杂 - 应命中 cloud
        ("我的身份证号 110...", {"privacy_level": "high"}),  # 隐私 - 强制 device
    ]

    for query, ctx in test_queries:
        result = router.route(query, ctx)
        print(f"Q: {query[:30]:<32} | tier={result.tier:<6} | "
              f"conf={result.confidence:.3f} | latency={result.latency_ms:.0f}ms")

