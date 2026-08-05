# ---
# chapter: 22
# topic: Agent 基础与工具调用
# topic_id: agent_tools.bidi_agent
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [strands-agents[bidi]]  # 真实运行需要；默认只运行框架无关 mock
# run: python 17_bidi_agent.py
# expected_runtime: 离线 <1s（mock）
# expected_output: 当前 Strands BidiAgent API 形状与一轮双向事件
# ---
# See: ../../../22_Agent基础与工具调用.md
# Interview hooks:
#   1. 什么叫“全双工”语音？和半双工（IVR）的本质区别？
#   2. BidiAgent 的 start/send/receive/stop 生命周期如何避免任务泄漏？
#   3. 为什么 experimental API 上生产前必须锁版本并做真实设备回归？
"""
Strands BidiAgent 双向事件流。

当前官方入口是 ``from strands.experimental.bidi import BidiAgent``，真实模型由
``strands.experimental.bidi.models`` 提供。由于该 API 仍标为 experimental，
本文件默认运行框架无关 mock；它只演示 start/send/receive/stop 的生命周期，
不伪造模型名、voice 参数、AudioConfig 或 start_session 回调接口。
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

OFFICIAL_API_SKETCH = """\
from strands import tool
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.models import BidiNovaSonicModel

@tool
def search_flight(origin: str, destination: str) -> str:
    return f"[demo] {origin} -> {destination}"

agent = BidiAgent(
    model=BidiNovaSonicModel(),
    tools=[search_flight],
    system_prompt="回答简洁；关键信息先确认。",
)
await agent.start()
await agent.send("帮我查明天北京到上海的航班")
async for event in agent.receive():
    handle(event)
await agent.stop()
"""


@dataclass(frozen=True)
class BidiEvent:
    """框架无关的教学事件；不是 Strands SDK 类型。"""

    type: str
    text: str


class MockBidiSession:
    """离线生命周期 mock；名称刻意不冒充 Strands ``BidiAgent``。"""

    def __init__(self) -> None:
        self.started = False
        self._input = ""

    async def start(self) -> None:
        self.started = True
        print("[mock] persistent connection started")

    async def send(self, text: str) -> None:
        if not self.started:
            raise RuntimeError("请先调用 start()")
        self._input = text
        print(f"用户: {text}")

    async def receive(self) -> AsyncIterator[BidiEvent]:
        if not self.started:
            raise RuntimeError("请先调用 start()")
        yield BidiEvent("tool_call", "search_flight(origin=北京, destination=上海)")
        yield BidiEvent("text", "已找到示例航班 CA1234，价格仅为 mock 数据。")
        yield BidiEvent("interrupt", "用户打断，停止当前输出")

    async def stop(self) -> None:
        self.started = False
        print("[mock] persistent connection stopped")


async def main() -> None:
    print("Official Strands surface:")
    print(OFFICIAL_API_SKETCH)

    session = MockBidiSession()
    await session.start()
    try:
        await session.send("帮我查明天北京到上海的航班")
        async for event in session.receive():
            print(f"{event.type}: {event.text}")
    finally:
        await session.stop()
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
