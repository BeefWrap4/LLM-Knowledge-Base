# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 面试真题 18-8：LangGraph Checkpoint 机制
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langgraph
# run: python 36_langgraph_checkpoint_demo.py
# expected_runtime: <1s
# expected_output: checkpoint config
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.7 (面试真题精选)
# Interview hooks:
#   1. LangGraph 的 Checkpoint 与传统数据库事务有什么异同？
#   2. "时间旅行调试"在 Agent 开发中能解决哪些痛点？
from langgraph.checkpoint.memory import MemorySaver

# MemorySaver 演示（实际使用可换成 PostgresSaver 等）
checkpointer = MemorySaver()
print("=== Checkpointer 类型 ===")
print(f"  type: {type(checkpointer).__name__}")

# 模拟一个简单的图配置
print("\n=== Checkpoint 核心价值 ===")
values = [
    "1. 断点续跑：执行到任意节点可暂停，之后从断点恢复",
    "2. Human-in-the-Loop：在关键节点暂停，等待人工审批后继续",
    "3. 时间旅行调试：可以回溯到历史任意状态",
    "4. 分支探索：从同一状态出发，探索不同路径（类似 Git 分支）",
    "5. 重放审计：完整记录执行轨迹，便于分析优化",
]
for v in values:
    print(f"  {v}")

# 使用 thread_id 隔离不同会话
config = {"configurable": {"thread_id": "conversation-1"}}
print("\n=== 多会话隔离 ===")
print(f"config: {config}")
print("graph.get_state_history(config) -> 状态历史列表")

if __name__ == "__main__":
    print("OK")
