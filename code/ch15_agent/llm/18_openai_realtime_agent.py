# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.3 OpenAI Realtime API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [websockets]  # 真实运行需要
# run: python 18_openai_realtime_agent.py
# expected_runtime: 离线 <1s（mock）；真实依赖 websockets + OpenAI 凭证
# expected_output: Realtime Agent 事件流结构演示
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.9.3-BidiAgent-与-Voice-Agent
# Interview hooks:
#   1. WebSocket + Server VAD 在 Realtime API 中如何协同？
#   2. input_audio_buffer.speech_started 事件触发后，客户端要做什么？(立即停止当前 TTS 播放)
#   3. 为什么 Realtime API 选择原生 PCM16 而不是 mp3？(减少编解码延迟)
"""
OpenAI Realtime API - 双向语音 Agent
通过 WebSocket 维持长连接，实时交换音频流
"""
import asyncio
import base64
import json
from typing import AsyncIterator


class OpenAIRealtimeAgent:
    """
    OpenAI Realtime API 客户端（mock 实现）

    协议：WebSocket over HTTPS 即 wss
    消息格式：JSON 事件流

    核心事件：
    - session.update: 配置会话
    - conversation.item.create: 添加对话项
    - response.audio.delta: 增量音频响应
    - input_audio_buffer.speech_started: 用户开始说话
    """

    REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(self, api_key: str, model: str = "gpt-realtime", mock: bool = True):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self.mock = mock
        self.session_config = None

    async def connect(self):
        if self.mock:
            print(f"[Mock connect] wss={self.REALTIME_URL}?model={self.model}")
            self.session_config = {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                },
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
            }
            return
        try:
            from websockets.client import connect
        except ImportError:
            raise RuntimeError("websockets not installed; install via `pip install websockets`")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        url = f"{self.REALTIME_URL}?model={self.model}"
        self.ws = await connect(url, extra_headers=headers)
        await self._send_event("session.update", {"session": self.session_config})

    async def send_audio(self, audio_chunk: bytes):
        """发送用户音频"""
        if self.mock:
            return
        await self._send_event("input_audio_buffer.append", {
            "audio": base64.b64encode(audio_chunk).decode()
        })

    async def listen(self) -> AsyncIterator[dict]:
        """监听服务器事件（mock 模式生成模拟事件）"""
        if self.mock:
            # 模拟一段会话流
            events = [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "response.audio.delta", "delta": "MOCK_AUDIO_CHUNK_1"},
                {"type": "response.audio.delta", "delta": "MOCK_AUDIO_CHUNK_2"},
                {"type": "response.audio_transcript.delta", "delta": "北京"},
                {"type": "response.audio_transcript.delta", "delta": "晴"},
                {"type": "conversation.item.created",
                 "item": {"type": "function_call",
                          "name": "get_weather",
                          "arguments": json.dumps({"city": "北京"})}},
            ]
            for e in events:
                yield self._normalize(e)
            return

        async for message in self.ws:
            event = json.loads(message)
            yield self._normalize(event)

    @staticmethod
    def _normalize(event: dict) -> dict:
        event_type = event.get("type")
        if event_type == "response.audio.delta":
            return {"type": "audio_chunk", "data": event.get("delta", "")}
        if event_type == "response.audio_transcript.delta":
            return {"type": "text", "data": event.get("delta", "")}
        if event_type == "input_audio_buffer.speech_started":
            return {"type": "user_started_speaking"}
        if event_type == "conversation.item.created":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                return {
                    "type": "tool_call",
                    "name": item["name"],
                    "args": json.loads(item["arguments"]),
                }
        if event_type == "error":
            return {"type": "error", "data": event.get("error")}
        return {"type": "other", "raw": event_type}

    async def _send_event(self, event_type: str, payload: dict):
        event = {"type": event_type, **payload}
        if self.ws is not None:
            await self.ws.send(json.dumps(event))


async def realtime_voice_demo():
    agent = OpenAIRealtimeAgent(api_key="sk-xxx", mock=True)
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
