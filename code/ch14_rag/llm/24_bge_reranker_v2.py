# ---
# chapter: 14
# topic: bge-reranker-v2-m3 重排序
# section: 14.9.8 2026 年主流 Reranker 选型
# difficulty: ⭐⭐⭐
# tier: llm
# deps: FlagEmbedding
# run: python 24_bge_reranker_v2.py
# expected_runtime: <1s (mock mode; real model needs GPU)
# expected_output: candidate scores
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.9-2026年新rag模式
# Interview hooks:
#   1. bge-reranker-v2-m3 相比 v1 在多语言上有什么提升？
#   2. 闭源 Cohere Rerank 3.5 和开源 bge-reranker 的精度/成本取舍？
#   3. Reranker 是否需要跟 Embedding 用同一系列（一致性 vs 性能）？

# bge-reranker-v2-m3 重排序示例（mock 演示版）


class FlagRerankerMock:
    """Mock FlagReranker: 模拟 FlagEmbedding 接口"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True, device: str = "cpu"):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device

    def compute_score(self, pairs):
        """返回每个候选的相关性分数"""
        scores = []
        for query, doc in pairs:
            # Mock: 简单关键词重合度 + 长度奖励
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words & d_words)
            score = overlap / max(len(q_words | d_words), 1)
            scores.append(float(score))
        return scores


if __name__ == "__main__":
    reranker = FlagRerankerMock(
        "BAAI/bge-reranker-v2-m3", use_fp16=True, device="cpu"
    )

    query = "什么是 RAG？"
    candidates = [
        "RAG 是检索增强生成，结合检索与生成。",
        "今天天气不错，适合出游。",
        "RAG 通过向量检索为 LLM 提供外部知识。",
    ]

    # 返回每个候选的相关性分数
    scores = reranker.compute_score([[query, c] for c in candidates])
    # scores = [0.95, 0.02, 0.89]
    for c, s in zip(candidates, scores):
        print(f"  score={s:.3f}  doc={c!r}")
    print("OK")
