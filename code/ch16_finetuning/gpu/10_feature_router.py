# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.7.3 基于查询特征的智能路由
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only)
# run: python 10_feature_router.py
# expected_runtime: <1s
# expected_output: 演示 8 条 query 经过关键词分类与规则路由后的目标 tier
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.7.3
# Interview hooks:
#   1. 基于规则的路由 vs 基于分类器路由的优缺点？
#   2. 默认 tier 设为 cloud 而不是 device 的工程原因（fail-safe）？
#   3. 如何用一个小分类器替代关键词匹配？训练数据如何收集？

"""
基于查询特征的智能路由 —— 适合特定业务场景
"""

from __future__ import annotations
from typing import Optional


# 路由规则: query_type -> tier
ROUTING_RULES = {
    # 简单问答 -> 端侧
    "greeting": "device",
    "faq": "device",
    "definition": "device",

    # 中等复杂度 -> 边缘
    "code_generation": "edge",
    "document_summary": "edge",
    "sql_query": "edge",

    # 复杂推理 -> 云端
    "multi_step_reasoning": "cloud",
    "math_proof": "cloud",
    "creative_writing": "cloud",
    "multi_modal": "cloud",

    # 隐私敏感 -> 端侧
    "personal_data": "device",
    "medical_query": "device",
}

# 关键词映射
KEYWORDS = {
    "greeting": ["你好", "hello", "hi", "早上好"],
    "math_proof": ["证明", "推导", "求解方程", "prove", "derive"],
    "code_generation": ["写代码", "function", "算法", "implement", "写一个函数"],
    "sql_query": ["sql", "select", "查询语句"],
    "creative_writing": ["写一首", "故事", "创作", "poem", "story"],
    "multi_step_reasoning": ["分析", "比较", "为什么", "analyze", "compare"],
    "personal_data": ["身份证", "住址", "电话", "银行卡"],
    "medical_query": ["症状", "诊断", "用药", "治疗", "医生"],
    "definition": ["什么是", "定义", "who is", "what is"],
}


class FeatureRouter:
    """基于查询特征的规则路由"""

    def route(self, query: str, query_type: Optional[str] = None) -> str:
        """根据查询特征选择部署层级"""
        # 自动分类（或使用独立分类器）
        if query_type is None:
            query_type = self._classify(query)

        # 默认 cloud（fail-safe: 复杂问题不漏到边缘）
        return ROUTING_RULES.get(query_type, "cloud")

    @staticmethod
    def _classify(query: str) -> str:
        """查询分类（关键词匹配）"""
        q_lower = query.lower()
        # 优先匹配更具体的类别
        priority = [
            "math_proof", "code_generation", "medical_query",
            "personal_data", "sql_query", "creative_writing",
            "multi_step_reasoning", "definition", "greeting",
        ]
        for qtype in priority:
            words = KEYWORDS.get(qtype, [])
            if any(w in q_lower for w in words):
                return qtype
        return "general"


if __name__ == "__main__":
    router = FeatureRouter()
    test_queries = [
        "你好",
        "什么是 Transformer?",
        "写一个 Python 排序函数",
        "证明 n^3-n 能被 6 整除",
        "我的身份证号是 110101199001011234",
        "写一首关于春天的诗",
        "分析中美关系发展",
        "普通查询无关键词",
    ]

    print(f"{'Query':<40} | Type           | Tier")
    print("-" * 75)
    for q in test_queries:
        qtype = router._classify(q)
        tier = router.route(q, qtype)
        print(f"{q:<40} | {qtype:<14} | {tier}")

    print("OK")
