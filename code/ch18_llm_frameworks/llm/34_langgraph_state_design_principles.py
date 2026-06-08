# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 面试真题 18-2：LangGraph State 设计原则
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langgraph
# run: python 34_langgraph_state_design_principles.py
# expected_runtime: <1s
# expected_output: state design demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.7 (面试真题精选)
# Interview hooks:
#   1. LangGraph 的 State 应该遵循哪些设计原则？
#   2. 为什么 State 必须可序列化？这与 checkpoint 有什么关系？
from typing import Annotated, TypedDict


# 模拟 add_messages 合并策略
def add_messages(a, b):
    return (a or []) + (b or [])


# ✅ 好的 State 设计
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史（追加）
    task_list: list[str]  # 待办任务
    completed: Annotated[set, lambda a, b: a | b]  # 完成的集合（并集）


# ❌ 避免的设计（仅作对比展示）
class BadState(TypedDict):
    everything: dict  # 大杂烩
    temp: object  # 不明确类型


# 演示
state: AgentState = {
    "messages": [{"role": "user", "content": "你好"}],
    "task_list": ["task1", "task2"],
    "completed": set(),
}

# 节点返回部分更新
update = {
    "messages": [{"role": "assistant", "content": "你好，有什么可以帮您？"}],
    "task_list": ["task2"],
    "completed": {"task1"},
}

# 合并：messages 追加，completed 取并集
state["messages"] = add_messages(state["messages"], update["messages"])
state["completed"] = state["completed"] | update["completed"]
print("合并后 State:")
print(f"  messages 数量: {len(state['messages'])}")
print(f"  completed: {state['completed']}")

if __name__ == "__main__":
    print("OK")
