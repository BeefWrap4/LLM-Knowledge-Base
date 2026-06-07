# ---
# chapter: 14
# topic: MRL (Matryoshka Representation Learning)
# section: 14.9.7 MRL Embeddings
# difficulty: ⭐⭐⭐
# tier: llm
# deps: openai (optional, mocked in demo)
# run: python 23_mrl_embeddings.py
# expected_runtime: <1s (mock mode)
# expected_output: variable-dimension embedding demo
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.9-2026年新rag模式
# Interview hooks:
#   1. MRL 训练的核心思想是什么（一个模型多种维度）？
#   2. 维度截断会带来多少精度损失？什么任务受影响最小？
#   3. MRL 和传统 Embedding 量化（PQ/SQ）有什么本质区别？

# MRL Embedding 使用（OpenAI text-embedding-3 系列原生支持，mock 演示版）


def get_mrl_embedding(text: str, dimensions: int = 512, api_key: str = None):
    """支持 256, 512, 1024, 3072 维动态选择"""
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=dimensions,
        )
        return response.data[0].embedding
    # Mock: 用确定种子生成对应维度向量
    import numpy as np
    rng = np.random.default_rng(hash(text) % (2**32))
    v = rng.normal(size=dimensions).astype("float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.tolist()


if __name__ == "__main__":
    text = "RAG 是检索增强生成"
    for dim in [256, 512, 1024, 3072]:
        vec = get_mrl_embedding(text, dimensions=dim, api_key=None)
        print(f"dimensions={dim:4d} -> vector len={len(vec)}")
