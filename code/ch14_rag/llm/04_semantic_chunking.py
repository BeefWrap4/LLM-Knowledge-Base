# ---
# chapter: 14
# topic: 语义分块
# section: 14.3.3 语义分块（Semantic Chunking）
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: sentence-transformers, numpy
# run: python 04_semantic_chunking.py
# expected_runtime: <1s (mock mode)
# expected_output: number of semantic chunks
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.3-文档处理与分块策略
# Interview hooks:
#   1. 语义分块和递归字符分块的本质区别是什么？
#   2. sliding window 在语义分块中起什么作用？
#   3. threshold_percentile 应该如何选择？调大/调小对结果有什么影响？
import re

import numpy as np


def semantic_chunking(
    text: str,
    embedder=None,                # SentenceTransformer 实例；None 时 mock
    window_size: int = 3,         # 滑动窗口大小
    threshold_percentile: float = 80,  # 断点阈值百分位
) -> list[str]:
    """
    语义分块：基于 Embedding 相似度检测语义断点

    原理：相邻句子如果语义相似度高（Embedding 余弦相似度高），应属于同一块；
         如果相似度骤降，说明发生了话题转换，应在此处断开。
    """
    # Step 1: 按句子分割
    sentences = re.split(r"(?<=[。！？;])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= window_size:
        return [text]

    # Step 2: 计算每个句子的 Embedding
    if embedder is not None:
        embeddings = embedder.encode(sentences, normalize_embeddings=True)
    else:
        # Mock: 用 sentence index 模拟 embedding，使相邻相似度低、制造断点
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(len(sentences), 16))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Step 3: 计算相邻窗口的相似度
    similarities = []
    for i in range(len(sentences) - window_size):
        # 窗口 A: [i, i+window_size)；窗口 B: [i+1, i+window_size+1)
        vec_a = np.mean(embeddings[i:i + window_size], axis=0)
        vec_b = np.mean(embeddings[i + 1:i + window_size + 1], axis=0)
        sim = float(np.dot(vec_a, vec_b))  # 余弦相似度（已归一化）
        similarities.append(sim)

    # Step 4: 检测断点（相似度低于阈值的点）
    threshold = np.percentile(similarities, 100 - threshold_percentile)
    breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]

    # Step 5: 按断点分块
    chunks = []
    start = 0
    for bp in breakpoints:
        end = bp + 1
        chunk = "".join(sentences[start:end])
        chunks.append(chunk)
        start = end
    chunks.append("".join(sentences[start:]))

    return chunks


if __name__ == "__main__":
    sample = (
        "今天天气很好。阳光明媚。适合出游。"
        "Python 是一种编程语言。它语法简洁。社区活跃。"
        "深度学习改变了很多行业。计算机视觉是典型应用。NLP 也是。"
    )
    chunks = semantic_chunking(sample, embedder=None, window_size=2, threshold_percentile=70)
    print(f"语义分块数: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {c}")
