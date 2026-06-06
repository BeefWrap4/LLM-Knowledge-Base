# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.2 VectorStoreIndex
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index, llama-index-embeddings-openai, llama-index-llms-openai
# run: python 13_llamaindex_vectorstore_index.py
# expected_runtime: <1s
# expected_output: query response
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.2
# Interview hooks:
#   1. VectorStoreIndex 的底层是如何将文本转为向量的？涉及哪些组件？
#   2. 为什么语义检索相比关键词检索有更好的"召回率"？
# VectorStoreIndex：向量索引（最常用）
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core import Settings

# 离线 mock 模式：定义最小的 embed 和 llm 替身
class _MockEmbed:
    def get_text_embedding(self, text):
        # 用长度作为简化的"向量"维度
        return [float(len(text))] * 8

class _MockLLM:
    def complete(self, prompt, **kwargs):
        return type("R", (), {"text": f"（mock）回答基于以下知识：{prompt[-200:]}..."})()

Settings.embed_model = _MockEmbed()
Settings.llm = _MockLLM()

# 加载文档（这里使用内存构造的 Document 列表，不依赖目录）
documents = [
    Document(text="公司退换货政策：自购买之日起7天内可无理由退换货，商品需保持原包装完整。"),
    Document(text="公司退换货政策：超过7天但仍在15天内，如商品有质量问题可申请换货。"),
    Document(text="公司的年假政策：工作满1年员工享受5天年假，满5年员工享受10天年假。"),
]

# 构建向量索引（自动分块+向量化+存储）
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("公司的退换货政策是什么？")
print(response)

if __name__ == "__main__":
    print("OK")
