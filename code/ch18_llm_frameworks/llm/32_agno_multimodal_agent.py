# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.8.7 Agno - ex-Phidata
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: agno (mocked structure)
# run: python 32_agno_multimodal_agent.py
# expected_runtime: <1s
# expected_output: agno config dump
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.8.7
# Interview hooks:
#   1. Agno 相比 LangGraph 的"极速启动"在实际工程中意味着什么？
#   2. 多模态 agent 的输入输出在协议层面如何统一？
"""
Agno 实战：多模态研究助手 - 离线 mock 结构
"""
import os

# 真实环境:
# from agno.agent import Agent
# from agno.models.openai import OpenAIChat
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.knowledge.pdf import PDFKnowledgeBase
# from agno.vectordb.pgvector import PgVector
# from agno.storage.sqlite import SqliteStorage

# 知识库 + 长期记忆（mock 配置）
knowledge_config = {
    "type": "PDFKnowledgeBase",
    "path": "./docs",
    "vector_db": "PgVector(table_name='agno_docs', db_url='postgresql://...')",
}
storage_config = {
    "type": "SqliteStorage",
    "table_name": "agent_sessions",
    "db_file": "sessions.db",
}

agent = {
    "name": "Researcher",
    "model": f"OpenAIChat(id='{os.environ.get('OPENAI_MODEL', 'gpt-5.6')}')",
    "tools": ["DuckDuckGoTools()"],
    "knowledge": knowledge_config,
    "storage": storage_config,
    "markdown": True,
    "show_tool_calls": True,
}


# 多模态输入：文本 + 图像
def mock_run(query: str, images=None) -> str:
    """模拟 agent.run 的输出"""
    if images:
        return f"（多模态）分析图像 {[PathImage.name for PathImage in images]} 并联网搜索：{query}"
    return f"（文本）{query}"


class _P:
    def __init__(self, name):
        self.name = name


print("=== Agno Agent 配置 ===")
for k, v in agent.items():
    print(f"  {k}: {v}")
print("\n启动时间: <5 微秒（benchmark）")
print(f"模型启动: {agent['model']}")
print(f"知识库: {agent['knowledge']['type']} -> {agent['knowledge']['path']}")

# 演示多模态调用
result = mock_run("分析这张图，并联网搜索相关最新研究", images=[_P("./chart.png")])
print(f"\n=== 多模态调用 ===\n{result}")

if __name__ == "__main__":
    print("OK")
