# ---
# chapter: 14
# topic: ColBERTv2 / PLAID
# section: 14.9.4 ColBERTv2 / PLAID（Late-Interaction 范式）
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: pylate
# run: python 22_colbertv2_plaid.py
# expected_runtime: requires GPU (real); <1s (mock)
# expected_output: PLAID index built, top-k scores
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.9-2026年新rag模式
# Interview hooks:
#   1. ColBERT 的 Max-Sim 打分为什么比 Bi-Encoder 点积更准？
#   2. PLAID 相对 ColBERT v1 解决了什么存储问题？
#   3. 什么场景下 ColBERTv2 比 Cross-Encoder 更合适？

# ColBERTv2 / PLAID 检索示例（mock 演示版）


class ColBERTMockModel:
    """Mock ColBERT: token-level 多向量"""

    def __init__(self, dim: int = 128, max_tokens: int = 32):
        self.dim = dim
        self.max_tokens = max_tokens

    def encode(self, texts, is_query: bool = False, show_progress_bar: bool = False):
        import numpy as np
        rng = np.random.default_rng(sum(map(ord, "".join(texts))) % (2**32) + int(is_query))
        out = []
        for t in texts:
            n = min(self.max_tokens, max(1, len(t.split()) + 1))
            v = rng.normal(size=(n, self.dim)).astype("float32")
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            out.append(v)
        return out


class PLAIDMockIndex:
    """Mock PLAID 索引"""

    def __init__(self, index_folder: str = "pylate_index"):
        self.index_folder = index_folder
        self.docs: list = []
        self.doc_embeddings: list = []

    def add_documents(self, documents, embeddings):
        self.docs = list(documents)
        self.doc_embeddings = list(embeddings)
        return self

    def retrieve(self, queries_embeddings, k: int = 5):
        import numpy as np
        results = []
        for q in queries_embeddings:
            scores = []
            for d in self.doc_embeddings:
                # Max-sim: sum_t max_d (q_t . d_d)
                s = 0.0
                for qt in q:
                    s += float(np.max(qt @ d.T))
                scores.append(s)
            order = np.argsort(scores)[::-1][:k]
            results.append([(int(i), float(scores[i])) for i in order])
        return results


if __name__ == "__main__":
    model = ColBERTMockModel()
    # 文档索引（每个文档产出多个 token-level 向量）
    documents = ["RAG 是检索增强生成", "ColBERT 是延迟交互模型"]
    documents_embeddings = model.encode(documents, is_query=False, show_progress_bar=False)
    print(f"Per-doc token embeddings: {[v.shape for v in documents_embeddings]}")

    # PLAID 索引（多向量 + 压缩）
    index = PLAIDMockIndex(index_folder="pylate_index")
    index.add_documents(documents=documents, embeddings=documents_embeddings)

    # 查询
    query_embeddings = model.encode(["什么是 RAG？"], is_query=True, show_progress_bar=False)
    scores = index.retrieve(queries_embeddings=query_embeddings, k=5)
    for q_idx, hits in enumerate(scores):
        print(f"\nQuery {q_idx} top-k:")
        for d_idx, s in hits:
            print(f"  doc={documents[d_idx]!r}  score={s:.3f}")
