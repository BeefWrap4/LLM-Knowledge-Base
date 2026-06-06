# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.2 SummaryIndex
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 14_llamaindex_summary_index.py
# expected_runtime: <1s
# expected_output: summary response
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.2
# Interview hooks:
#   1. SummaryIndex 适合什么场景？tree_summarize 模式如何工作？
#   2. SummaryIndex 与 VectorStoreIndex 在性能与效果上如何权衡？
from llama_index.core import SummaryIndex, Document
from llama_index.core import Settings

class _MockLLM:
    def complete(self, prompt, **kwargs):
        return type("R", (), {"text": "（mock）所有文档的核心观点摘要：聚焦文档索引与检索。"})()

Settings.llm = _MockLLM()

documents = [
    Document(text="文档1: 关于 LangChain 的核心组件与 LCEL 编程范式。"),
    Document(text="文档2: LangGraph 提供状态图与多 Agent 协作能力。"),
    Document(text="文档3: LlamaIndex 专注数据索引与 RAG 检索增强生成。"),
]

# SummaryIndex 适合需要对整个文档集合进行概括的场景
summary_index = SummaryIndex.from_documents(documents)

query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"  # 递归摘要模式
)
response = query_engine.query("总结所有文档的核心观点")
print(response)

if __name__ == "__main__":
    print("OK")
