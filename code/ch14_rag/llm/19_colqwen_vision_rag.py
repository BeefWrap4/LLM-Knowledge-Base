# ---
# chapter: 14
# topic: ColQwen Vision-RAG
# section: 14.9.1 Vision-RAG：ColPali / ColQwen
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: colpali_engine, torch, Pillow
# run: python 19_colqwen_vision_rag.py
# expected_runtime: requires GPU + ~3GB model download (real); <1s (mock)
# expected_output: doc and query multi-vector embeddings (mock)
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.9-2026年新rag模式
# Interview hooks:
#   1. ColPali/ColQwen 为什么能"跳过 OCR"？核心思想是什么？
#   2. 多向量 + Max-Sim 延迟交互相比点积有什么优势？
#   3. 文档图像 Embedding 的存储成本如何优化（PLAID 压缩）？

# ColQwen Vision-RAG 简化示例（mock 演示版，避免强制下载模型）
# 真实运行需 GPU + colpali_engine：
#     pip install colpali_engine torch Pillow


class ColQwenMockModel:
    """Mock ColQwen2：模拟多向量输出"""

    def __init__(self, patch_dim: int = 128, token_dim: int = 64, n_patches: int = 1024):
        self.patch_dim = patch_dim
        self.token_dim = token_dim
        self.n_patches = n_patches

    def encode_images(self, images):
        import numpy as np

        rng = np.random.default_rng(len(images))
        return rng.normal(size=(len(images), self.n_patches, self.patch_dim)).astype("float32")

    def encode_queries(self, queries):
        import numpy as np

        rng = np.random.default_rng(len(queries) + 7)
        return rng.normal(size=(len(queries), self.token_dim, self.patch_dim)).astype("float32")

    def score_multi_vector(self, query_emb, doc_emb):
        # Max-sim: [B_q, Q_tok, D] vs [B_d, P, D] -> [B_q, B_d]
        # 简化: dot 然后 sum (真实是 max-sim)
        return (query_emb @ doc_emb.transpose(0, 2, 1)).sum(axis=1)


if __name__ == "__main__":
    model = ColQwenMockModel()
    # 1. 加载视觉-文档编码器 (mock)
    print("Loading ColQwen2 (mock)...")

    # 2. 编码文档图像（每页 PDF 转 PNG）
    page_images = [f"page_{i}.png" for i in range(10)]
    doc_embeddings = model.encode_images(page_images)
    print(f"Doc embeddings: {doc_embeddings.shape}  # [B_pages, P_patches, D_dim]")

    # 3. 编码查询
    query_embeddings = model.encode_queries(["RAG 的核心思想是什么？"])
    print(f"Query embeddings: {query_embeddings.shape}  # [1, Q_tokens, D_dim]")

    # 4. 延迟交互打分（max-sim 算子）
    scores = model.score_multi_vector(query_embeddings, doc_embeddings)
    top_k_indices = scores[0].topk(3).indices.tolist() if hasattr(scores[0], "topk") else [0, 1, 2]
    print(f"Top-3 pages: {top_k_indices}")
