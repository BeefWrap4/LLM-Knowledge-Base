# ---
# chapter: 14
# topic: Late Chunking (jina.ai)
# section: 14.9.3 Late Chunking
# difficulty: ⭐⭐⭐
# tier: llm
# deps: requests
# run: python 21_late_chunking.py
# expected_runtime: <1s (mock; real needs JINA_API_KEY)
# expected_output: late-chunked embeddings list
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.9-2026年新rag模式
# Interview hooks:
#   1. Late Chunking 相比传统 chunking 的核心差异是什么（顺序 vs 倒序）？
#   2. 为什么必须用 Long-Context Embedding 模型才能做 Late Chunking？
#   3. Late Chunking 与 Contextual Retrieval 的取舍（计算时机 vs 注入开销）？

# jina Late Chunking 示例（mock 演示版，避免强制调用 API）


def late_chunking(document: str, chunk_size: int = 512, api_key: str = None):
    """
    Late Chunking: 整篇文档 Embedding 后按 token 位置切分
    """
    if api_key:
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "jina-embeddings-v3",
                "input": [document],  # 整篇文档
                "task": "retrieval.passage",
                "late_chunking": True,  # 开启 late chunking
                "embedding_dim": 1024,
            },
        )
        # 返回 [N_chunks, D] 矩阵
        return response.json()["data"][0]["embeddings"]
    # Mock: 模拟 N_chunks 个 1024 维向量
    n_chunks = max(1, len(document) // chunk_size)
    import numpy as np

    rng = np.random.default_rng(hash(document) % (2**32))
    return rng.normal(size=(n_chunks, 1024)).astype("float32").tolist()


if __name__ == "__main__":
    doc = "RAG 是检索增强生成。" * 100  # 长文档
    chunks = late_chunking(doc, chunk_size=512, api_key=None)
    if isinstance(chunks, list):
        print(f"Late Chunking 产出 {len(chunks)} 个 chunk 向量, dim={len(chunks[0])}")
