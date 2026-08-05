# ---
# chapter: 24
# topic: Agent 工作流编排与多智能体
# topic_id: llm_frameworks.langgraph_human_in_the_loop
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langgraph
# run: python 12_langgraph_human_in_the_loop.py
# expected_runtime: <1s
# expected_output: approval state demo
# ---
# See: ../../../24_Agent工作流编排与多智能体.md
# Interview hooks:
#   1. LangGraph 的 interrupt() 与传统"暂停-恢复"机制有什么不同？
#   2. 哪些业务场景必须使用 Human-in-the-Loop？设计原则是什么？
"""
LangGraph Human-in-the-Loop 实战
关键操作审批流：生成操作计划 → 暂停等待人工确认 → 继续执行
"""

from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    approved: bool


def planner_node(state: ApprovalState) -> dict:
    """生成执行计划"""
    plan = """
    执行计划：
    1. 备份数据库
    2. 更新模型版本到 v2.1
    3. 运行回归测试
    4. 切换流量到新版本
    """
    return {"plan": plan, "messages": [{"role": "assistant", "content": f"已生成计划：\n{plan}"}]}


def human_approval_node(state: ApprovalState) -> dict:
    """人机协同关键节点：使用 interrupt() 暂停执行"""
    # 真实场景：interrupt() 会暂停图执行，等待外部输入
    # user_decision = interrupt(f"请审批以下计划：\n{state['plan']}\n\n输入 'approve' 或 'reject':")
    # 离线模拟：直接根据本地变量
    user_decision = "approve"  # 模拟人工输入
    if user_decision.lower() == "approve":
        return {
            "approved": True,
            "messages": [{"role": "assistant", "content": "计划已批准，开始执行。"}],
        }
    else:
        return {"approved": False, "messages": [{"role": "assistant", "content": "计划被拒绝。"}]}


def execute_node(state: ApprovalState) -> dict:
    """执行已批准的计划"""
    return {"messages": [{"role": "assistant", "content": "执行完成：所有步骤已成功。"}]}


def route_after_approval(state: ApprovalState) -> Literal["execute", END]:
    return "execute" if state["approved"] else END


# 构建图
builder = StateGraph(ApprovalState)
builder.add_node("planner", planner_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("execute", execute_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "human_approval")
builder.add_conditional_edges("human_approval", route_after_approval)
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=MemorySaver())

# 第一轮：直接调用至 execute 节点
config = {"configurable": {"thread_id": "approval-001"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "准备部署新版本"}], "approved": False}, config=config
)

print("=== 执行结果 ===")
for msg in result["messages"]:
    if isinstance(msg, dict):
        print(f"AI: {msg.get('content', '')[:200]}")
    else:
        print(f"AI: {msg.content[:200]}")

if __name__ == "__main__":
    print("OK")
