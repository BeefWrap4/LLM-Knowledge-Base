# ---
# chapter: 25
# topic: 可恢复 Agent 运行时
# topic_id: agent_tools.durable_execution
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 20_durable_execution.py
# expected_runtime: ~4s（mock 中每次 await asyncio.sleep 模拟耗时）
# expected_output: 第二次运行会跳过已完成的步骤（演示断点续传）
# ---
# See: ../../../25_可恢复Agent运行时.md
# Interview hooks:
#   1. Durable Execution 适合所有 Agent 吗？成本代价是什么？
#   2. 事件溯源（Event Sourcing）和"任务状态持久化"有何区别？
#   3. 步骤完成的事件和 checkpoint 字段分别承担什么语义？
"""
Pydantic AI Durable Execution - 持久化执行示例
核心思想：每个步骤产生事件，事件持久化后可重放
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ResearchStep:
    """研究流程中的一个步骤"""

    step_id: str
    name: str
    status: str = "pending"
    result: str = ""
    started_at: str = ""
    completed_at: str = ""


class PostgresEventStore:
    """简化的 Postgres 事件存储（实际中用真实 DB）"""

    def __init__(self, connection_string: str, table_name: str = "agent_events"):
        self.connection_string = connection_string
        self.table_name = table_name
        self._in_memory: dict[str, list] = {}

    async def append_event(
        self, task_id: str, event_type: str, payload: dict, checkpoint: dict = None
    ) -> None:
        """追加事件"""
        event = {
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
            "checkpoint": checkpoint,
            "timestamp": datetime.now().isoformat(),
        }
        self._in_memory.setdefault(task_id, []).append(event)
        # 实际实现中：INSERT INTO events ...

    async def load_events(self, task_id: str) -> list:
        """加载任务的所有历史事件"""
        return self._in_memory.get(task_id, [])


async def search_basic_info(topic: str) -> str:
    await asyncio.sleep(0.1)
    return f"基础信息: {topic} 的入门介绍..."


async def deep_analysis(topic: str, context: dict) -> str:
    await asyncio.sleep(0.1)
    return f"深度分析: {topic} 的核心机制..."


async def write_report(topic: str, context: dict) -> str:
    await asyncio.sleep(0.1)
    return f"完整报告: {topic} 的研究报告..."


async def run_research_task(task_id: str, topic: str, event_store: PostgresEventStore) -> list:
    """
    完整的耐久执行流程：
    1. 检查是否有未完成的事件（恢复）
    2. 如果没有，从头开始
    3. 每步都持久化事件
    """
    history = await event_store.load_events(task_id)
    completed_ids: set = set()
    if history:
        print(f"[恢复] 任务 {task_id}，已执行 {len(history)} 个事件")
        completed_ids = {e["payload"].get("step_id") for e in history if e["event_type"] == "step_completed"}

    steps = [
        ResearchStep(step_id="1", name="搜索基础信息"),
        ResearchStep(step_id="2", name="深度分析"),
        ResearchStep(step_id="3", name="撰写报告"),
    ]
    results: list = []

    for step in steps:
        if step.step_id in completed_ids:
            print(f"[跳过] 步骤 {step.step_id} 已完成")
            step.status = "completed"
            results.append(step)
            continue

        await event_store.append_event(
            task_id=task_id,
            event_type="step_started",
            payload={"step_id": step.step_id, "name": step.name},
        )
        step.status = "running"
        step.started_at = datetime.now().isoformat()

        try:
            if step.step_id == "1":
                step.result = await search_basic_info(topic)
            elif step.step_id == "2":
                step.result = await deep_analysis(topic, {})
            elif step.step_id == "3":
                step.result = await write_report(topic, {})

            step.status = "completed"
            step.completed_at = datetime.now().isoformat()

            await event_store.append_event(
                task_id=task_id,
                event_type="step_completed",
                payload={"step_id": step.step_id, "result": step.result},
                checkpoint={"step_id": step.step_id, "status": "completed"},
            )
            results.append(step)
            print(f"[完成] 步骤 {step.step_id}: {step.result[:50]}...")

        except Exception as e:
            await event_store.append_event(
                task_id=task_id,
                event_type="step_failed",
                payload={"step_id": step.step_id, "error": str(e)},
            )
            raise

    return results


async def main():
    store = PostgresEventStore(connection_string="postgresql://mock")
    task_id = "task-001"

    print("=== 第一次运行（cold start） ===")
    await run_research_task(task_id, "Python GIL", store)

    print("\n=== 第二次运行（恢复） ===")
    await run_research_task(task_id, "Python GIL", store)
    print("\nOK")


if __name__ == "__main__":
    asyncio.run(main())
