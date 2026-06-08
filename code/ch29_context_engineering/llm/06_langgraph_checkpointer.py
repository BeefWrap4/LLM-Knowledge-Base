# ---
# chapter: 29
# topic: LangGraph MemorySaver — 持久化 state 解决长会话 context 丢失
# section: 29.4.2
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: langgraph (optional)
# run: python 06_langgraph_checkpointer.py
# expected_runtime: <1s (mock)
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.4.2
# Cross-refs:
#   - Ch15 Agent (ReAct + State)
#   - Ch18 LangGraph 核心概念
#   - Ch20 LLMOps (state 审计)
#
# Interview hooks:
#   - "LangGraph 持久化解决什么问题?"    →  LLM context 截断后, 旧消息仍可从 checkpointer 恢复
#   - "MemorySaver vs PostgresSaver?"    →  内存版用于 dev/test; Postgres/Redis 用于 production
#   - "thread_id 的作用?"                →  同一会话的所有 state 共享一个 thread, 跨 turn 累积

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """LangGraph-style state — 跨 turn 累积, 关键载体。"""

    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    turn: int = 0


class MemorySaver:
    """Mock LangGraph MemorySaver — 按 thread_id 存 state 历史。"""

    def __init__(self):
        self._threads: dict[str, list[AgentState]] = {}
        self._latest: dict[str, AgentState] = {}

    def get(self, config: dict) -> AgentState | None:
        return self._latest.get(config["configurable"]["thread_id"])

    def put(self, config: dict, state: AgentState) -> None:
        tid = config["configurable"]["thread_id"]
        self._threads.setdefault(tid, []).append(state)
        self._latest[tid] = state

    def history(self, thread_id: str) -> list[AgentState]:
        return list(self._threads.get(thread_id, []))


# 节点: 模拟 LLM 节点 (向 state 追加消息)
def call_model(state: AgentState, user_input: str) -> AgentState:
    state.messages.append({"role": "user", "content": user_input})
    # mock 助手回复
    state.messages.append({"role": "assistant", "content": f"[mock] echo: {user_input[:40]}"})
    state.turn += 1
    return state


def should_continue(state: AgentState) -> str:
    return "end" if state.turn >= 3 else "agent"


def build_graph() -> tuple[Callable, MemorySaver]:
    """模拟 LangGraph 编译: 节点 + 条件边 + checkpointer。"""
    saver = MemorySaver()
    config = {"configurable": {"thread_id": "user-42"}}

    def run_step(user_input: str) -> AgentState:
        # 1. 从 checkpointer 恢复 state
        state = saver.get(config) or AgentState()
        # 2. 调用模型
        state = call_model(state, user_input)
        # 3. 决策
        nxt = should_continue(state)
        # 4. 写回
        saver.put(config, state)
        return state

    return run_step, saver


def run_demo() -> None:
    run_step, saver = build_graph()

    print("=== LangGraph MemorySaver 持久化演示 ===\n")
    queries = [
        "你好, 我叫 Alice, 喜欢看科幻片",
        "推荐一部最近上映的?",
        "我之前说过喜欢什么类型?",
    ]
    for q in queries:
        st = run_step(q)
        print(f"[turn {st.turn}] user: {q}")
        print(f"         assistant: {st.messages[-1]['content']}")

    print("\n=== Checkpointer 中累积的 state 历史 ===")
    for i, snap in enumerate(saver.history("user-42"), 1):
        print(f"snapshot {i}: turn={snap.turn}  messages={len(snap.messages)}")

    print("\n关键: 即使 LLM 上下文窗口截断旧消息, 完整 state 仍在 checkpointer 中, 可按需加载。")
    print("生产环境用 PostgresSaver / RedisSaver 替代 MemorySaver。")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
