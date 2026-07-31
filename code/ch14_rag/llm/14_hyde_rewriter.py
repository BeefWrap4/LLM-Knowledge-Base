# ---
# chapter: 14
# topic: Query Rewriting & HyDE
# section: 14.5.3 Query Rewriting（查询重写）
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: sentence-transformers
# run: python 14_hyde_rewriter.py
# expected_runtime: <1s (mock mode)
# expected_output: hypothetical document embedding demo
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.5-检索与重排序
# Interview hooks:
#   1. HyDE 的核心洞察是什么？为什么生成文档的 embedding 比查询 embedding 检索更准？
#   2. Query Rewriting 的三种策略（同义词扩展/HyDE/子查询分解）适用场景有何不同？
#   3. HyDE 失败的情况有哪些（短查询对模型不熟悉的话题）？

import os

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6")
OPENAI_REASONING_KWARGS = (
    {"reasoning_effort": "none"} if DEFAULT_OPENAI_MODEL.startswith("gpt-5.6") else {}
)


# Query Rewriting 策略

# 1. 同义词扩展
REWRITE_TEMPLATE_SYNONYM = """
将用户查询扩展为多个语义等价的查询，覆盖不同表达方式。

用户查询：{query}

请输出 3 个语义等价但表述不同的查询（每行一个，不要编号）：
"""

# 2. 伪文档扩展（HyDE - Hypothetical Document Embedding）
REWRITE_TEMPLATE_HYDE = """
请根据用户查询，生成一段可能包含答案的理想文档片段。
这段文档将用于语义检索，请尽可能包含相关的关键词和概念。

用户查询：{query}

理想文档片段：
"""

# 3. 子查询分解（用于复杂多步问题）
REWRITE_TEMPLATE_SUBQUERY = """
将复杂查询分解为多个简单子查询。

复杂查询：{query}

请分解为 2-3 个可以独立回答的子查询（每行一个）：
"""


# HyDE 实现
class HyDERewriter:
    """
    HyDE（Hypothetical Document Embedding）：
    用 LLM 生成假想的理想文档，然后用这个文档的 Embedding 去检索
    核心洞察：生成文档的 Embedding 比查询 Embedding 更"丰富"
    """

    def __init__(self, llm_client=None, embedder=None):
        self.llm = llm_client
        self.embedder = embedder

    def rewrite(self, query: str):
        # 生成假想文档
        prompt = REWRITE_TEMPLATE_HYDE.format(query=query)
        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                **OPENAI_REASONING_KWARGS,
            )
            hypothetical_doc = response.choices[0].message.content
        else:
            hypothetical_doc = f"[Mock] 关于「{query}」的假想文档片段。"

        # 返回假想文档的 Embedding（而非原始查询的）
        if self.embedder is not None:
            return self.embedder.encode(hypothetical_doc, normalize_embeddings=True)
        # Mock embedding
        import numpy as np

        rng = np.random.default_rng(hash(query) % 2**32)
        vec = rng.normal(size=64).astype("float32")
        vec /= np.linalg.norm(vec)
        return vec


if __name__ == "__main__":
    hyde = HyDERewriter(llm_client=None, embedder=None)
    q = "什么是 RAG？"
    emb = hyde.rewrite(q)
    print(f"查询: {q}")
    print(f"假想文档 embedding shape: {emb.shape}, norm: {float((emb**2).sum()):.3f}")
    print("OK")
