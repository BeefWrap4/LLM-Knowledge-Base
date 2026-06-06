# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.2.2 核心概念 State, Node, Edge
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langgraph, langchain-openai
# run: python 10_langgraph_core_concepts.py
# expected_runtime: <1s
# expected_output: state + node + edge demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.2.2
# Interview hooks:
#   1. LangGraph 的 StateGraph 与普通 Chain 的本质区别是什么？
#   2. add_messages 与 operator.add 在 Annotated 中有什么区别？
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

# ===== 1. 定义 State（状态）- 在节点间传递的共享数据 =====
class AgentState(TypedDict):
    """
    State 是 LangGraph 的核心：定义了图中流动的数据结构。
    每个 Node 接收 State，返回 State 的部分更新。
    """
    messages: Annotated[list, add_messages]  # add_messages：追加而非覆盖
    next_step: str  # 用于条件路由

# ===== 2. 定义 Node（节点）- 执行具体逻辑的函数 =====
@tool
def search_tool(q: str) -> str:
    """模拟搜索"""
    return f"搜索结果：{q}"

@tool
def calculator_tool(expr: str) -> str:
    """计算器"""
    return str(eval(expr, {"__builtins__": {}}, {}))

# 工具列表
tools = [search_tool, calculator_tool]

def agent_node(state: AgentState) -> dict:
    """
    Agent 节点：调用 LLM 决定下一步行动
    绑定工具后 LLM 可以返回 function_call
    """
    # 实际场景会绑定工具：llm_with_tools = llm.bind_tools(tools)
    # response = llm_with_tools.invoke(state["messages"])
    return {"messages": [{"role": "assistant", "content": "（模拟 LLM 决定下一步）"}]}

# ===== 3. 定义 Edge（边）- 控制流向 =====
# 普通边（无条件）
# graph.add_edge("node_a", "node_b")  # A 执行后总是到 B

# 条件边（根据状态决定）
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    条件路由函数：
    如果 LLM 返回了 function_call → 执行工具
    否则 → 结束
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, dict) and last_message.get("tool_calls"):
        return "tools"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# 构造一个最小可运行的 LangGraph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": END, "end": END})
graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "你好"}], "next_step": ""})
print("图执行结果:", result)

if __name__ == "__main__":
    print("OK")
