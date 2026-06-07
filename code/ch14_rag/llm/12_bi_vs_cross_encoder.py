# ---
# chapter: 14
# topic: Bi-Encoder vs Cross-Encoder
# section: 14.5.2 Re-ranking（重排序）
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy, sentence-transformers
# run: python 12_bi_vs_cross_encoder.py
# expected_runtime: <1s (mock mode)
# expected_output: side-by-side similarity comparison
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.5-检索与重排序
# Interview hooks:
#   1. Bi-Encoder 和 Cross-Encoder 在精度/速度上的本质权衡是什么？
#   2. 为什么不能用 Cross-Encoder 直接做全量召回？
#   3. 两阶段架构（召回+精排）相比单阶段有什么优势？

# Bi-Encoder：分别编码查询和文档，点积计算相似度
# 优点：可预先计算文档向量，搜索速度快
# 缺点：查询和文档没有交互，精度有限

import numpy as np


def bi_encoder_demo():
    """Mock 演示 Bi-Encoder 的 dot-product 相似度"""
    rng = np.random.default_rng(0)
    # 假设 bi_encoder.encode() 返回归一化向量
    query_embedding = rng.normal(size=64)
    query_embedding /= np.linalg.norm(query_embedding)
    doc_embedding = rng.normal(size=64)
    doc_embedding /= np.linalg.norm(doc_embedding)
    similarity = np.dot(query_embedding, doc_embedding)
    return float(similarity)


def cross_encoder_demo():
    """Mock 演示 Cross-Encoder 的拼接打分"""
    # 真实场景: cross_encoder.predict("[CLS] q [SEP] d") -> scalar
    rng = np.random.default_rng(1)
    return float(rng.random())


# Cross-Encoder：将查询和文档拼接后一起编码
# 优点：查询和文档在注意力层充分交互，精度高
# 缺点：无法预计算，每次都要完整前向传播
def cross_encoder_pair_score(query: str, doc: str) -> float:
    pair_input = f"[CLS] {query} [SEP] {doc}"
    # score = cross_encoder.predict(pair_input)  # 真实使用
    score = cross_encoder_demo()
    return score


if __name__ == "__main__":
    bi_sim = bi_encoder_demo()
    ce_score = cross_encoder_pair_score(
        "什么是 RAG？",
        "RAG 是一种将检索和生成结合的技术...",
    )
    print(f"Bi-Encoder dot-product 相似度: {bi_sim:.4f}")
    print(f"Cross-Encoder 相关性分数: {ce_score:.4f}")
