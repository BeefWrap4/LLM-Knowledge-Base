# ---
# chapter: 19
# topic: RAG 数据解析、分块与索引
# topic_id: rag_indexing.embedding_similarity
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy, sentence-transformers
# run: python 06_embedding_similarity.py
# expected_runtime: <1s (mock mode without model download)
# expected_output: 3x3 similarity matrix
# ---
# See: ../../../19_RAG数据解析分块与索引.md
# Interview hooks:
#   1. 余弦相似度和点积的关系是什么？什么情况下等价？
#   2. 为什么 Embedding 后通常要做 L2 归一化？
#   3. 维度更高的 Embedding 一定更好吗？成本如何权衡？

import numpy as np


# Embedding 相似度计算示例（mock 演示版，避免下载模型）
def get_mock_embeddings(sentences: list[str]) -> np.ndarray:
    """Mock: 用确定性哈希制造"语义相似"的向量"""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(len(sentences), 16))
    # 让前两句方向接近，第三句方向不同
    if len(sentences) >= 3:
        base[1] = base[0] + 0.05 * rng.normal(size=16)
        base[2] = rng.normal(size=16)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    return base / norms


def main() -> None:
    sentences = [
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的一种方法",
        "苹果是一种水果",
    ]
    embeddings = get_mock_embeddings(sentences)

    # 计算余弦相似度矩阵
    sim_matrix = np.dot(embeddings, embeddings.T)
    print("相似度矩阵：")
    print(np.round(sim_matrix, 2))
    # 期望效果：前两句相似度高（都是 AI 相关），第三句差异大
    print("OK")


if __name__ == "__main__":
    main()
