# ---
# chapter: 14
# topic: IVF 倒排索引
# section: 14.4.5 IVF 索引原理
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: faiss-cpu, numpy
# run: python 09_ivf_index.py
# expected_runtime: <1s
# expected_output: IVF index built and queried
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.4-embedding-与向量数据库
# Interview hooks:
#   1. IVF 索引的 nlist 应该如何选择（4*sqrt(N) ~ 16*sqrt(N)）？
#   2. nprobe 调大/调小对召回率、速度有什么影响？
#   3. IVF 与 HNSW 各自的最佳适用场景？

import numpy as np


def create_ivf_index(vectors: np.ndarray, nlist: int = 100):
    """
    IVF 索引构建

    参数：
    - nlist: 聚类中心数量（通常 4*sqrt(N) ~ 16*sqrt(N)）
    - nprobe: 查询时搜索的单元数（越大精度越高，速度越慢）
    """
    try:
        import faiss
    except ImportError:
        print("[Mock] faiss 未安装，返回 None；pip install faiss-cpu 启用真实索引。")
        return None

    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)  # 用于聚类的精确索引
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    # IVF 需要先训练
    index.train(vectors)
    index.add(vectors)
    # 搜索参数：探索多少个聚类单元
    index.nprobe = 10  # 默认 1，增大提升召回率
    return index


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(2000, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = create_ivf_index(vectors, nlist=32)
    if index is not None:
        D, I = index.search(vectors[:1], 5)
        print(f"Top-5 邻居: {I[0].tolist()}, 距离: {D[0].tolist()}")
    print("OK")
