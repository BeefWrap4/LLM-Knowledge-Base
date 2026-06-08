# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.2.3 多步骤 Research Agent
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langgraph, langchain-openai
# run: python 11_langgraph_research_agent.py
# expected_runtime: <1s
# expected_output: graph build success + simple invoke
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.2.3
# Interview hooks:
#   1. LangGraph 中条件边（add_conditional_edges）的映射字典起什么作用？
#   2. MemorySaver 持久化机制如何与 thread_id 配合实现多会话隔离？
"""
LangGraph 实战：多步骤 Research Agent

工作流：接收问题 → 搜索 → 分析 → 判断是否需要更多搜索
   如果需要 → 继续搜索（循环）
   如果足够 → 生成最终答案
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# ===== 定义工具 =====
@tool
def web_search(query: str) -> str:
    """搜索网络获取信息"""
    # 模拟搜索结果
    mock_results = {
        "langgraph": "LangGraph 是 LangChain 团队开发的状态图Agent框架，支持循环、条件分支和人机协同。",
        "react": "ReAct 是 Reasoning+Acting 的缩写，是 LLM Agent 的经典范式。",
        "multi-agent": "多Agent系统通过多个Agent协作完成复杂任务，常见框架有 AutoGen 和 CrewAI。",
    }
    for k, v in mock_results.items():
        if k in query.lower():
            return v
    return f"关于'{query}'的搜索结果：这是一个活跃的研究领域..."


@tool
def analyze_data(data: str) -> str:
    """分析数据并提取关键洞察"""
    return f"分析结果：'{data}' 中包含3个关键点，建议进一步研究第2点。"


# ===== 定义 State =====
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_topic: str
    search_count: int
    analysis_complete: bool


# ===== 定义 Nodes =====
tools = [web_search, analyze_data]


def researcher_node(state: ResearchState) -> dict:
    """研究员节点：决定搜索什么"""
    topic = state.get("research_topic", "unknown")
    sys_msg = SystemMessage(content=f"你是一个研究助手。当前主题：{topic}。使用 search 工具获取信息。")
    return {
        "messages": [{"role": "assistant", "content": f"研究主题：{topic}"}],
        "search_count": state.get("search_count", 0),
    }


def analyst_node(state: ResearchState) -> dict:
    """分析师节点：分析已收集的信息"""
    count = state.get("search_count", 0)
    return {
        "messages": [{"role": "assistant", "content": "分析完成"}],
        "search_count": count + 1,
        "analysis_complete": count >= 2,  # 简化：搜索 2 次后完成
    }


def writer_node(state: ResearchState) -> dict:
    """撰稿节点：生成最终研究报告"""
    return {"messages": [{"role": "assistant", "content": "最终报告：研究完成。"}]}


# ===== 条件路由 =====
def route_after_analyst(state: ResearchState) -> Literal["researcher", "writer"]:
    if state.get("analysis_complete", False):
        return "writer"
    return "researcher"


def route_after_researcher(state: ResearchState) -> Literal["tools", "analyst"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "analyst"


# ===== 构建图 =====
def build_research_graph():
    builder = StateGraph(ResearchState)

    # 添加节点
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)

    # 添加边
    builder.set_entry_point("researcher")

    # 条件边
    builder.add_conditional_edges(
        "researcher", route_after_researcher, {"tools": "analyst", "analyst": "analyst"}
    )
    builder.add_conditional_edges(
        "analyst", route_after_analyst, {"researcher": "researcher", "writer": "writer"}
    )
    builder.add_edge("writer", END)

    # 编译（带持久化）
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ===== 运行 =====
graph = build_research_graph()

config = {"configurable": {"thread_id": "research-001"}}
result = graph.invoke(
    {
        "messages": [{"role": "user", "content": "请研究LangGraph的最新特性"}],
        "research_topic": "LangGraph",
        "search_count": 0,
        "analysis_complete": False,
    },
    config=config,
)

# 打印执行结果
for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(f"AI: {msg.content[:200]}...")
    elif isinstance(msg, dict) and msg.get("content"):
        print(f"AI: {msg['content'][:200]}...")

if __name__ == "__main__":
    print("OK")
