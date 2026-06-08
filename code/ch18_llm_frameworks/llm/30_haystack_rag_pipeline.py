# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.8.5 Haystack 2.x
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: haystack-ai (mocked structure)
# run: python 30_haystack_rag_pipeline.py
# expected_runtime: <1s
# expected_output: pipeline structure
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.8.5
# Interview hooks:
#   1. Haystack 2.x 的"Component 协议"相比 LangChain 的 Chain 抽象有什么不同？
#   2. Hayhooks 一键部署为 MCP Server 对生产有什么价值？
"""
Haystack 2.x 实战：context-engineered RAG pipeline - 离线 mock 结构
"""
# 真实环境:
# from haystack import Pipeline, component, Document
# from haystack.components.builders import ChatPromptBuilder
# from haystack.components.generators.chat import OpenAIChatGenerator
# from haystack.components.retrievers import InMemoryBM25Retriever
# from haystack.document_stores.in_memory import InMemoryDocumentStore
# from hayhooks import deploy


# ===== 1. 自定义 rerank 组件 =====
class ContextualCompressor:
    """模拟 @component 装饰的类"""

    def run(self, documents, query):
        # 简化版：截断过短 / 过长文档
        compressed = [d for d in documents if 50 < len(d.content) < 2000]
        return {"documents": compressed}


# ===== 2. 模拟 Document =====
class Document:
    def __init__(self, content, meta=None):
        self.content = content
        self.meta = meta or {}


# ===== 3. 构造 pipeline 拓扑 =====
pipeline_components = {
    "retriever": {
        "type": "InMemoryBM25Retriever",
        "store": "InMemoryDocumentStore",
    },
    "compressor": {
        "type": "ContextualCompressor",
        "description": "截断 50-2000 字符的文档",
    },
    "prompt_builder": {
        "type": "ChatPromptBuilder",
        "template": (
            "Given context and answer the question.\n"
            "Context: {% for d in documents %}{{ d.content }}\\n{% endfor %}\n"
            "Question: {{query}}"
        ),
    },
    "llm": {
        "type": "OpenAIChatGenerator",
        "model": "gpt-4o",
    },
}

pipeline_connections = [
    ("retriever.documents", "compressor.documents"),
    ("compressor.documents", "prompt_builder.documents"),
    ("prompt_builder.prompt", "llm.messages"),
]

print("=== Haystack 2.x Pipeline 拓扑 ===")
for name, cfg in pipeline_components.items():
    print(f"  [{name}] type={cfg['type']}")
print("\n连接:")
for src, dst in pipeline_connections:
    print(f"  {src} → {dst}")

# 模拟 compressor
compressor = ContextualCompressor()
docs = [Document("a" * 30), Document("a" * 100), Document("a" * 3000)]
result = compressor.run(docs, query="test")
print("\n=== Compressor 演示 ===")
print("输入: 3 篇文档（30/100/3000 字符）")
print(f"输出: {len(result['documents'])} 篇文档（保留 50-2000 字符）")

print("\n=== Hayhooks 部署（生产）===")
print("deploy(pipe, name='my_rag', mcp_server=True)  # 暴露为 MCP 工具")

if __name__ == "__main__":
    print("OK")
