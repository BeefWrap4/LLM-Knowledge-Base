# ---
# chapter: 14
# topic: RAG 作为 MCP Tool
# section: 14.7.2 MCP 与 RAG 的工程化集成
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: mcp (optional, mocked in demo)
# run: python 17_rag_mcp_tool.py
# expected_runtime: <1s (mock mode)
# expected_output: MCP tool definition + handle_tool_call demo
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.7-rag-与-agent-的融合
# Interview hooks:
#   1. 2026 年视角下，RAG 在 Agent 架构中的定位是什么？
#   2. RAG 通过 MCP 暴露有什么工程价值（统一接口、可观测性）？
#   3. MCP 工具的 inputSchema 应该包含哪些关键字段？filters 为什么重要？

# 2026年：RAG 作为 MCP Tool 的架构（mock 演示版，不强依赖 mcp 库）


class RAGMCPTool:
    """
    将 RAG 系统封装为 MCP Tool —— 2026年工程标准

    这样任何支持 MCP 的 Agent（Claude、GPT、自研 Agent）
    都可以通过统一接口调用 RAG 能力
    """

    def __init__(self, vectorstore=None, embedder=None, top_k: int = 5):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = top_k

    async def handle_tool_call(self, name: str, arguments: dict) -> list:
        """MCP Tool 调用入口"""
        query = arguments.get("query", "")
        filters = arguments.get("filters", {})  # 元数据过滤

        # 执行检索
        if self.vectorstore is not None and hasattr(self.vectorstore, "similarity_search"):
            docs = self.vectorstore.similarity_search(
                query,
                k=self.top_k,
                filter=filters,
            )
        else:
            # Mock 返回
            class MockDoc:
                def __init__(self, content, meta):
                    self.page_content = content
                    self.metadata = meta

            docs = [
                MockDoc(
                    f"关于「{query}」的检索结果 {i + 1}（filter={filters}）",
                    {"source": f"mock_doc_{i + 1}.md", "score": 0.9 - i * 0.1},
                )
                for i in range(self.top_k)
            ]

        # 格式化返回（MCP 标准格式）
        results = []
        for i, doc in enumerate(docs, 1):
            text = (
                f"[{i}] 来源: {doc.metadata.get('source', 'unknown')}\n"
                f"相关度: {doc.metadata.get('score', 'N/A')}\n"
                f"内容: {doc.page_content[:500]}"
            )
            # 真实环境返回 [TextContent(type="text", text=...)]
            results.append({"type": "text", "text": text})
        return results

    def get_tool_definition(self) -> dict:
        """MCP Tool 定义 —— 让 Agent 知道何时调用 RAG"""
        return {
            "name": "knowledge_base_search",
            "description": (
                "从企业知识库中检索相关信息。"
                "适用于：公司政策、技术文档、产品手册、历史数据等内部知识查询。"
                "当问题涉及公司内部信息时使用此工具。"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询"},
                    "filters": {
                        "type": "object",
                        "description": "可选的元数据过滤条件（source/tag/date）",
                    },
                },
                "required": ["query"],
            },
        }


if __name__ == "__main__":
    import asyncio

    tool = RAGMCPTool(vectorstore=None, embedder=None, top_k=3)
    print("Tool Definition:")
    import json

    print(json.dumps(tool.get_tool_definition(), ensure_ascii=False, indent=2))
    print("\nTool Call Result:")
    result = asyncio.run(
        tool.handle_tool_call(
            "knowledge_base_search",
            {"query": "年假政策", "filters": {"source": "hr.pdf"}},
        )
    )
    for r in result:
        print("---")
        print(r["text"])
