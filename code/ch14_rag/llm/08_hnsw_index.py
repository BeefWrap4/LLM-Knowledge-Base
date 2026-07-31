# ---
# chapter: 14
# topic: HNSW 索引参数调优
# section: 14.4.4 HNSW 索引原理
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: faiss-cpu, numpy
# run: python 08_hnsw_index.py
# expected_runtime: <1s
# expected_output: HNSW index built and queried (mock vectors)
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.4-embedding-与向量数据库
# Interview hooks:
#   1. HNSW 搜索 O(log N) 的复杂度是如何通过"多层图+贪心下降"实现的？
#   2. M、efConstruction、efSearch 三个参数如何影响精度/速度/内存？
#   3. HNSW 为什么不适合超大规模（亿级）？此时应该用什么索引？

# HNSW 参数调优
import numpy as np


def create_hnsw_index(vectors: np.ndarray, m: int = 32, ef_construction: int = 200):
    """
    创建 HNSW 索引

    参数说明：
    - M: 每个节点的最大连接数（越大图越稠密，精度↑内存↑）
    - efConstruction: 构建时的搜索深度（越大构建越慢，精度↑）
    - efSearch: 搜索时的搜索深度（越大搜索越慢，精度↑）
    """
    try:
        import faiss
    except ImportError:
        print("[Mock] faiss 未安装，返回 None；pip install faiss-cpu 启用真实索引。")
        return None

    dim = vectors.shape[1]
    # 使用 Inner Product（需要向量已归一化，等价于余弦相似度）
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    # 添加向量（训练+添加）
    index.add(vectors)

    # 搜索时可调整 efSearch 平衡速度和精度
    index.hnsw.efSearch = 128  # 默认 16，增大可提升召回率

    return index


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(1000, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = create_hnsw_index(vectors, m=16, ef_construction=100)
    if index is not None:
        D, I = index.search(vectors[:1], 5)
        print(f"Top-5 邻居: {I[0].tolist()}, 距离: {D[0].tolist()}")
    print("OK")
