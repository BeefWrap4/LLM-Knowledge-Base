# ---
# chapter: 29
# topic: Haystack 2.x 基础 RAG Pipeline (检索 + 提示 + 生成)
# section: 29.8
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: haystack-ai
# run: python 04_haystack_basic_pipeline.py
# expected_runtime: <2s (mock 后端)
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.8
# Cross-refs:
#   - Ch14 RAG (检索层)
#   - Ch18 LLM Frameworks (Haystack vs LangChain)
#   - Ch13 Prompt Engineering (模板)
#
# Interview hooks:
#   - "Haystack 2.x 核心思想?"  →  组件化 Pipeline, 节点+边, 显式数据流
#   - "Haystack vs LangChain?"  →  Haystack 偏 production pipeline, LangChain 偏通用框架
#   - "Pipeline 如何连接组件?"  →  pipe.connect("node_a.output", "node_b.input")

from __future__ import annotations

# 保证 standalone 运行: 优先尝试真实 haystack, 失败则用 mock
try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    HAS_HAYSTACK = True
except Exception:
    HAS_HAYSTACK = False


def run_real() -> None:
    doc_store = InMemoryDocumentStore()
    docs = [
        Document(content="Haystack 2.x 是 deepset 推出的 LLM 应用框架, 强调组件化 pipeline。"),
        Document(content="Context Engineering 关注模型每步推理时看到的全部信息。"),
        Document(content="LangGraph 是 LangChain 推出的有状态 agent 编排框架, 支持持久化 checkpoint。"),
        Document(content="Sub-agent 模式让每个子任务拥有独立 context, 避免污染。"),
    ]
    doc_store.write_documents(docs)
    retriever = InMemoryBM25Retriever(document_store=doc_store)

    template = """根据下列文档回答用户问题。如果文档中无答案, 请回答"未知"。

{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

问题: {{ question }}
回答:"""
    prompt_builder = PromptBuilder(template=template)

    pipe = Pipeline()
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.connect("retriever.documents", "prompt_builder.documents")

    result = pipe.run(
        {
            "retriever": {"query": "什么是 Context Engineering"},
            "prompt_builder": {"question": "什么是 Context Engineering"},
        }
    )
    print("Pipeline 渲染后的 prompt (截断):")
    print(result["prompt_builder"]["prompt"][:300] + "...")


def run_mock() -> None:
    """Mock 模式: 模拟 Haystack Pipeline 的数据流, 不安装 haystack。"""

    # 极简 Pipeline 抽象
    class MockComponent:
        def __init__(self, name, fn):
            self.name, self.fn = name, fn

        def run(self, **kwargs):
            return self.fn(**kwargs)

    class MockPipeline:
        def __init__(self):
            self.comps, self.edges = {}, []

        def add_component(self, name, comp):
            self.comps[name] = comp

        def connect(self, src, dst):
            self.edges.append((src, dst))

        def run(self, inputs):
            print("  [mock] 模拟管道执行:")
            print(f"  [mock] 边: {self.edges}")
            print(f"  [mock] 输入: { {k: type(v).__name__ for k, v in inputs.items()} }")
            return {
                "prompt_builder": {
                    "prompt": (
                        "根据下列文档回答用户问题。\n"
                        "- Haystack 2.x 强调组件化 pipeline\n"
                        "- Context Engineering 关注模型每步推理时看到的全部信息\n"
                        "问题: 什么是 Context Engineering\n回答:"
                    )
                }
            }

    docs = [
        "Haystack 2.x 是 deepset 推出的 LLM 应用框架",
        "Context Engineering 关注模型每步推理时看到的全部信息",
    ]
    retriever = MockComponent(
        "retriever", lambda query: {"documents": [type("D", (), {"content": c}) for c in docs]}
    )
    pb = MockComponent("prompt_builder", lambda documents, question: {"prompt": "(略)"})

    pipe = MockPipeline()
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", pb)
    pipe.connect("retriever.documents", "prompt_builder.documents")

    result = pipe.run(
        {
            "retriever": {"query": "什么是 Context Engineering"},
            "prompt_builder": {"question": "什么是 Context Engineering"},
        }
    )
    print("Mock Pipeline 渲染后的 prompt:")
    print(result["prompt_builder"]["prompt"])


if __name__ == "__main__":
    print("=== Haystack 2.x 基础 Pipeline ===\n")
    if HAS_HAYSTACK:
        print("[real] 检测到 haystack, 运行真实管道")
        run_real()
    else:
        print("[mock] 未检测到 haystack, 运行 mock 管道 (pip install haystack-ai 启用真实模式)")
        run_mock()
    print("\nOK")
