# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.3 查询引擎与聊天引擎
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 17_llamaindex_query_chat_engine.py
# expected_runtime: <1s
# expected_output: query + chat engine demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.3
# Interview hooks:
#   1. RetrieverQueryEngine 与直接使用 index.as_query_engine() 有什么不同？
#   2. SimilarityPostprocessor 的 similarity_cutoff 如何影响检索结果？
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings

class _MockEmbed:
    def get_text_embedding(self, text):
        return [float(len(text))] * 8

class _MockLLM:
    def complete(self, prompt, **kwargs):
        return type("R", (), {"text": "（mock）查询结果。"})()
    def chat(self, messages, **kwargs):
        return type("R", (), {"message": type("M", (), {"content": "（mock）聊天回复。"})()})()

Settings.llm = _MockLLM()
Settings.embed_model = _MockEmbed()

# 加载并构建索引
documents = [
    Document(text="公司2025年的营收目标：实现年度营收 100 亿元，同比增长 25%。"),
    Document(text="公司的核心价值观：客户第一、员工成长、持续创新。"),
    Document(text="招聘原则：价值观匹配 + 能力胜任 + 发展潜力。"),
]
index = VectorStoreIndex.from_documents(documents)

# ===== 查询引擎：无状态 =====
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.0)],  # mock 下用 0.0 兜底
)

response = query_engine.query("2025年的营收目标是多少？")
print(f"查询结果: {response}")
print(f"参考来源: {[n.metadata for n in response.source_nodes]}")

# ===== 聊天引擎：有状态（带记忆） =====
# 离线 mock：直接用字符串模拟 chat_engine
print("\n=== 多轮对话模拟 ===")
print("第1轮: 公司的核心价值观是什么？ -> 客户第一、员工成长、持续创新。")
print("第2轮（需理解'这个'指代）: 这个价值观如何体现在招聘中？ -> 价值观匹配 + 能力胜任 + 发展潜力。")
print("第3轮: 能否用一句话总结？ -> 以客户为中心，注重员工成长与创新。")

if __name__ == "__main__":
    print("OK")
