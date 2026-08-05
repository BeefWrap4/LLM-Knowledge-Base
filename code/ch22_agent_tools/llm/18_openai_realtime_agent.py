# ---
# chapter: 22
# topic: Agent 基础与工具调用
# topic_id: agent_tools.openai_realtime_agent
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [websockets]  # 仅 LLM_MOCK=0 的真实运行需要
# run: python 18_openai_realtime_agent.py
# expected_runtime: 离线 <1s（默认 mock）；真实运行需显式 LLM_MOCK=0
# expected_output: 当前 Realtime GA 会话与事件流结构演示，最后输出 OK
# ---
# See: ../../../22_Agent基础与工具调用.md
# Interview hooks:
#   1. WebSocket + Server VAD 在 Realtime API 中如何协同？
#   2. input_audio_buffer.speech_started 事件触发后，客户端要做什么？(立即停止当前 TTS 播放)
#   3. WebSocket 客户端为何必须消费 response.output_audio.delta？（done 事件不含音频字节）
"""
OpenAI Realtime API - 双向语音 Agent
通过服务端 WebSocket 维持长连接，实时交换音频流。

默认 LLM_MOCK=1，只生成可 JSON 序列化的协议事件，不导入 WebSocket 客户端、
不读取 API Key、也不建立网络连接。真实运行必须显式设置 LLM_MOCK=0。
"""

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator

DEFAULT_REALTIME_MODEL = "gpt-realtime-2.1"


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


class OpenAIRealtimeAgent:
    """
    OpenAI Realtime API 服务端 WebSocket 客户端。

    协议：WebSocket over TLS（wss）
    消息格式：JSON 事件流

    核心事件：
    - session.update: 配置会话
    - conversation.item.create: 添加对话项
    - response.output_audio.delta: 增量音频响应
    - response.output_audio_transcript.delta: 增量音频转写
    - input_audio_buffer.speech_started: 用户开始说话
    - response.done: 完整响应；工具调用参数位于 response.output
    """

    REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        mock: bool | None = None,
    ):
        self.api_key = api_key
        self.model = model or os.getenv("OPENAI_REALTIME_MODEL", DEFAULT_REALTIME_MODEL)
        self.ws = None
        # 程序参数不能绕过全局安全门禁：只有精确 LLM_MOCK=0 才允许真实连接。
        self.mock = os.environ.get("LLM_MOCK") != "0" or bool(mock)
        self.session_config = self._build_session_config()

    def _build_session_config(self) -> dict:
        """构造 Realtime GA 的嵌套会话结构。"""
        return {
            "type": "realtime",
            "model": self.model,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm"},
                    "voice": "marin",
                },
            },
            "instructions": "用简洁、自然的中文回答；调用工具前确认必要参数。",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": "auto",
        }

    def session_update_event(self) -> dict:
        """返回可直接发送、也便于离线测试的 session.update 事件。"""
        return {"type": "session.update", "session": self.session_config}

    @staticmethod
    def _auth_headers(api_key: str) -> dict[str, str]:
        """GA Realtime 不再需要 OpenAI-Beta 请求头。"""
        return {"Authorization": f"Bearer {api_key}"}

    async def connect(self):
        if self.mock:
            print(f"[Mock connect] wss={self.REALTIME_URL}?model={self.model}")
            return

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("真实 Realtime 运行需要 LLM_MOCK=0 与 OPENAI_API_KEY")
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            raise RuntimeError("websockets not installed; install via `pip install websockets`")

        url = f"{self.REALTIME_URL}?model={self.model}"
        self.ws = await connect(url, additional_headers=self._auth_headers(api_key))
        await self._send(self.session_update_event())

    async def send_audio(self, audio_chunk: bytes):
        """发送用户音频"""
        if self.mock:
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_chunk).decode("ascii"),
            }
        )

    @staticmethod
    def mock_server_events() -> list[dict]:
        """返回与当前服务器事件名一致的离线事件流。"""
        return [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.output_audio.delta", "delta": "TU9DS19BVURJTw=="},
            {"type": "response.output_audio_transcript.delta", "delta": "北京晴"},
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "status": "completed",
                            "name": "get_weather",
                            "call_id": "call_mock_weather",
                            "arguments": json.dumps({"city": "北京"}, ensure_ascii=False),
                        }
                    ],
                },
            },
        ]

    async def listen(self) -> AsyncIterator[dict]:
        """监听服务器事件（mock 模式生成模拟事件）"""
        if self.mock:
            for e in self.mock_server_events():
                yield self._normalize(e)
            return

        if self.ws is None:
            raise RuntimeError("请先调用 connect()")
        async for message in self.ws:
            event = json.loads(message)
            yield self._normalize(event)

    @staticmethod
    def _normalize(event: dict) -> dict:
        event_type = event.get("type")
        if event_type == "response.output_audio.delta":
            return {"type": "audio_chunk", "data": event.get("delta", "")}
        if event_type in {
            "response.output_audio_transcript.delta",
            "response.output_text.delta",
        }:
            return {"type": "text", "data": event.get("delta", "")}
        if event_type == "input_audio_buffer.speech_started":
            return {"type": "user_started_speaking"}
        if event_type == "response.done":
            output = event.get("response", {}).get("output", [])
            for item in output:
                if item.get("type") != "function_call":
                    continue
                try:
                    args = json.loads(item.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {"_raw": item.get("arguments", "")}
                return {
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "call_id": item.get("call_id", ""),
                    "args": args,
                }
            return {"type": "response_done", "status": event.get("response", {}).get("status")}
        if event_type == "error":
            return {"type": "error", "data": event.get("error")}
        return {"type": "other", "raw": event_type}

    async def _send(self, event: dict):
        if self.ws is None:
            raise RuntimeError("请先调用 connect()")
        await self.ws.send(json.dumps(event, ensure_ascii=False))


async def realtime_voice_demo():
    agent = OpenAIRealtimeAgent()
    await agent.connect()

    async for event in agent.listen():
        etype = event.get("type")
        if etype == "audio_chunk":
            pass  # 实际播放音频
        elif etype == "text":
            print(f"助手说: {event['data']}")
        elif etype == "user_started_speaking":
            print("[用户开始说话，停止当前播放]")
        elif etype == "tool_call":
            print(f"[工具调用] {event['name']}({event['args']})")
        elif etype == "error":
            print(f"[错误] {event['data']}")
        else:
            print(f"[other] {event}")
    print("OK")


if __name__ == "__main__":
    asyncio.run(realtime_voice_demo())
