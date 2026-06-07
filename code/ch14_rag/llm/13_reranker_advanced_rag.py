# ---
# chapter: 14
# topic: Cross-Encoder 重排序 + 完整 Advanced RAG
# section: 14.5.2 Re-ranking（重排序）
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: sentence-transformers
# run: python 13_reranker_advanced_rag.py
# expected_runtime: <1s (mock mode)
# expected_output: reranked results with mock scores
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.5-检索与重排序
# Interview hooks:
#   1. Rerank 阶段为什么不直接用更大的 embedding 模型做召回？
#   2. recall_k 和 final_k 如何选择（典型 20-50 / 5-10）？
#   3. bge-reranker-large 相比 MiniLM 有什么精度/速度权衡？

# Cross-Encoder Re-ranking 实战
class Reranker:
    """Cross-Encoder 重排序器（mock 演示版）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        推荐模型：
        - BAAI/bge-reranker-large：中文场景首选
        - BAAI/bge-reranker-base：速度优先
        - cross-encoder/ms-marco-MiniLM-L-6-v2：英文场景
        """
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, documents: list[str], top_k: int = 5):
        """
        对候选文档进行重排序

        Args:
            query: 用户查询
            documents: 召回的候选文档列表
            top_k: 返回 Top-K

        Returns:
            [(文档, 重排序分数), ...]
        """
        if self.model is None:
            # Mock: 关键词重合度打分
            q_words = set(query.lower().split())
            scored = []
            for doc in documents:
                d_words = set(doc.lower().split())
                score = float(len(q_words & d_words)) / max(len(q_words | d_words), 1)
                scored.append((doc, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


# 完整 RAG Pipeline 集成 Re-ranking
class AdvancedRAG:
    """带混合搜索 + 重排序的 Advanced RAG"""

    def __init__(self, vectorstore, retriever, reranker, llm_client=None):
        self.vectorstore = vectorstore
        self.retriever = retriever      # 混合检索器
        self.reranker = reranker        # Cross-Encoder 重排序
        self.llm = llm_client

    def query(self, question: str, recall_k: int = 20, final_k: int = 5) -> dict:
        # Step 1: 混合检索，召回更多候选
        if hasattr(self.retriever, "embedder"):
            query_embedding = self.retriever.embedder.encode(question)
        else:
            import numpy as np
            query_embedding = np.random.default_rng(0).normal(size=64).astype("float32")
        recalled = self.retriever.search(
            question, query_embedding, top_k=recall_k
        )
        candidate_docs = [self.retriever.documents[i] for i, _ in recalled]

        # Step 2: Cross-Encoder 重排序
        reranked = self.reranker.rerank(question, candidate_docs, top_k=final_k)

        # Step 3: 取 Top 文档构建上下文
        context = "\n\n---\n\n".join([
            f"[相关度 {score:.3f}] {doc[:500]}"
            for doc, score in reranked
        ])

        # Step 4: LLM 生成
        prompt = f"""基于以下检索结果回答问题：

{context}

---

问题：{question}

请给出准确、简洁的回答。"""
        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content
        else:
            answer = f"[Mock] 基于 {len(reranked)} 条精排结果回答: {question}"

        return {
            "answer": answer,
            "sources": reranked,
            "recall_count": len(recalled),
        }


if __name__ == "__main__":
    # 简单 mock 演示
    docs = ["什么是 RAG", "今天天气", "RAG 检索增强生成", "机器学习"]
    import numpy as np
    emb = np.random.default_rng(0).normal(size=(len(docs), 64)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    from importlib import import_module
    # 复用 11_hybrid_retriever 的 HybridRetriever
    hr_mod = import_module("11_hybrid_retriever")
    retriever = hr_mod.HybridRetriever(docs, emb)
    reranker = Reranker(model_name="mock")
    rag = AdvancedRAG(vectorstore=None, retriever=retriever, reranker=reranker, llm_client=None)
    out = rag.query("RAG 是什么", recall_k=4, final_k=2)
    print(f"answer: {out['answer']}")
    print(f"recall_count: {out['recall_count']}")
    for doc, score in out["sources"]:
        print(f"  score={score:.3f}  doc={doc!r}")
