# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.8.3 多模态 RAG - MaxSim 检索评分教学骨架
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: numpy
# run: python 11_multimodal_rag.py
# expected_runtime: <1s
# expected_output: 确定性多向量 MaxSim 排名
# ---
# See: ../tutorial/21_多模态大模型.md#21-7-3-实战：构建多模态rag系统
# Interview hooks:
#   1. late interaction 为什么保留 query/document token 级向量？
#   2. MaxSim 的计算量、索引体积与召回质量如何权衡？
#   3. 完整视觉 RAG 还需要哪些编码、溯源、生成与评估步骤？

import numpy as np


def maxsim_score(query_vectors: np.ndarray, document_vectors: np.ndarray) -> float:
    """按 query token 求最大相似度后求和；输入应由真实编码器归一化。"""
    if query_vectors.ndim != 2 or document_vectors.ndim != 2:
        raise ValueError("query_vectors and document_vectors must both be 2-D")
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValueError("query/document embedding dimensions must match")
    similarities = query_vectors @ document_vectors.T
    return float(similarities.max(axis=1).sum())


def main() -> None:
    # 确定性教学向量，只验证评分与排序；不是 ColPali 输出，也不是端到端问答。
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = {
        "page-0": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        "page-1": np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32),
    }
    ranking = sorted(
        ((page, maxsim_score(query, vectors)) for page, vectors in documents.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    assert ranking[0][0] == "page-0"
    print("[STRUCTURE ONLY] No PDF, vision encoder, query encoder, reranker, or VLM was used.")
    print(ranking)


if __name__ == "__main__":
    main()
    print("OK")
