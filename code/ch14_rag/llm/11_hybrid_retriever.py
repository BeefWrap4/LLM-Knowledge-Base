# ---
# chapter: 14
# topic: 混合搜索 (向量 + BM25 + RRF)
# section: 14.5.1 混合搜索（Hybrid Search）
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: rank_bm25, faiss-cpu, numpy
# run: python 11_hybrid_retriever.py
# expected_runtime: <1s
# expected_output: hybrid search result ranking
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.5-检索与重排序
# Interview hooks:
#   1. 为什么纯向量检索对精确 ID / 型号 / 缩写词表现不佳？
#   2. RRF (Reciprocal Rank Fusion) 公式为什么用 1/(k+rank) 而不是线性加权？
#   3. 向量检索和 BM25 的权重 alpha/beta 如何调节？

# 混合搜索实现
import numpy as np


class HybridRetriever:
    """
    混合检索器：向量相似度 + BM25 融合
    """

    def __init__(self, documents: list[str], embeddings: np.ndarray, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.embeddings = embeddings  # 向量 [N, D]

        # BM25 初始化
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            BM25Okapi = None
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b) if BM25Okapi else None

        # 向量检索用 FAISS
        try:
            import faiss
        except ImportError:
            faiss = None
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim) if faiss is not None else None  # Inner Product = 余弦（已归一化）
        if self.index is not None:
            self.index.add(embeddings)

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        alpha: float = 0.5,  # 向量权重
        beta: float = 0.5,   # BM25 权重
    ) -> list[tuple[int, float]]:
        """
        混合搜索 + RRF 融合
        """
        # 向量检索 Top-K
        if self.index is not None:
            vector_scores, vector_indices = self.index.search(
                query_embedding.reshape(1, -1), k=min(top_k * 2, len(self.documents))
            )
            vector_scores = vector_scores[0]
            vector_indices = vector_indices[0]
        else:
            vector_indices = np.array([], dtype=np.int64)

        # BM25 检索 Top-K
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        else:
            bm25_top_indices = np.array([], dtype=np.int64)

        # RRF（Reciprocal Rank Fusion）融合
        rrf_k = 60
        rrf_scores: dict[int, float] = {}

        # 向量检索排名贡献
        for rank, idx in enumerate(vector_indices):
            if idx < 0:  # FAISS 返回 -1 表示不够结果
                break
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + alpha / (rrf_k + rank + 1)

        # BM25 排名贡献
        for rank, idx in enumerate(bm25_top_indices):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + beta / (rrf_k + rank + 1)

        # 按 RRF 分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# 使用示例
if __name__ == "__main__":
    docs = [
        "RAG 检索增强生成。",
        "Vector search is fast for semantic match.",
        "BM25 用于关键词匹配。",
        "今天天气很好。",
        "GPT-4 是 OpenAI 的大模型。",
    ]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(docs), 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q_emb = emb[0]  # 用 doc0 自身的向量查询自己

    retriever = HybridRetriever(docs, emb)
    results = retriever.search("什么是 RAG", q_emb, top_k=3, alpha=0.7, beta=0.3)
    for idx, score in results:
        print(f"  idx={idx}  score={score:.4f}  doc={docs[idx]!r}")
