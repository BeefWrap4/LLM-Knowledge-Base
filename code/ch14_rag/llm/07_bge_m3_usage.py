# ---
# chapter: 14
# topic: BGE-M3 多向量检索
# section: 14.4.2 Embedding 模型选型
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: FlagEmbedding
# run: python 07_bge_m3_usage.py
# expected_runtime: <1s (mock mode; real model requires GPU)
# expected_output: dict of dense/sparse/colbert vectors
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.4-embedding-与向量数据库
# Interview hooks:
#   1. BGE-M3 相比 bge-large-zh 的核心优势是什么？
#   2. dense / sparse / colbert 三种向量各自的检索用途是什么？
#   3. 多向量检索（ColBERT）相对双塔 Bi-Encoder 解决了什么问题？

# BGE-M3 使用示例（支持多向量检索）
# 生产环境使用真实模型（需 ~2GB 下载 + GPU）：
#     from FlagEmbedding import BGEM3FlagModel
#     model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
#     embeddings = model.encode(sentences, return_dense=True, return_sparse=True, return_colbert_vecs=True)


def bge_m3_mock_encode(sentences: list[str]):
    """Mock 演示 dense/sparse/colbert 三种向量的结构"""
    import numpy as np

    rng = np.random.default_rng(0)
    dense = rng.normal(size=(len(sentences), 1024)).astype(np.float32)
    sparse = [{f"tok_{i}": float(rng.random()) for i in range(5)} for _ in sentences]
    colbert = [rng.normal(size=(8, 1024)).astype(np.float32) for _ in sentences]
    return {
        "dense_embeddings": dense,
        "sparse_embeddings": sparse,
        "colbert_vecs": colbert,
    }


if __name__ == "__main__":
    sentences = ["什么是机器学习？", "Machine learning is..."]
    out = bge_m3_mock_encode(sentences)
    print("dense_embeddings shape:", out["dense_embeddings"].shape)
    print("sparse_embeddings[0] keys:", list(out["sparse_embeddings"][0].keys()))
    print("colbert_vecs[0] shape:", out["colbert_vecs"][0].shape)
    print("OK")
