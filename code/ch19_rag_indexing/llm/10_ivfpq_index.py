# ---
# chapter: 19
# topic: RAG 数据解析、分块与索引
# topic_id: rag_indexing.ivfpq_index
# difficulty: ⭐⭐⭐
# tier: llm
# deps: faiss-cpu, numpy
# run: python 10_ivfpq_index.py
# expected_runtime: <1s
# expected_output: IVFPQ index built and queried
# ---
# See: ../../../19_RAG数据解析分块与索引.md
# Interview hooks:
#   1. PQ 量化如何把高维向量压缩到低维？压缩率如何计算？
#   2. PQ 的精度损失主要来自哪里？有什么缓解策略？
#   3. 什么时候适合用 IVF+PQ 而不是 HNSW？

import numpy as np


def create_ivfpq_index(vectors: np.ndarray, nlist: int = 100, m: int = 16, nbits: int = 8):
    """
    IVF + PQ 组合索引

    参数：
    - m: 将向量分成 m 个子向量（m 必须整除 dim）
    - nbits: 每个子量化的比特数（通常 8）

    内存节省：原始 dim*32bit → m*8bit，压缩率约 dim*4/m
    """
    try:
        import faiss
    except ImportError:
        print("[Mock] faiss 未安装，返回 None；pip install faiss-cpu 启用真实索引。")
        return None

    dim = vectors.shape[1]
    # 要求 m 整除 dim，否则 IVFPQ 训练会失败
    if dim % m != 0:
        m = max(1, dim // 4)
    quantizer = faiss.IndexFlatIP(dim)
    # PQ 参数: m=子向量数, nbits=每个子向量量化比特
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 10
    return index


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    # 注意: dim 必须能整除 m
    vectors = rng.normal(size=(2000, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # 小样本 smoke 使用 4 bit（每个子空间 16 个质心），避免 8 bit 需要更多训练向量。
    index = create_ivfpq_index(vectors, nlist=32, m=8, nbits=4)
    if index is not None:
        D, I = index.search(vectors[:1], 5)
        print(f"Top-5 邻居: {I[0].tolist()}, 距离: {D[0].tolist()}")
    print("OK")
