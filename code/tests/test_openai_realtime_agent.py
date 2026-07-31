"""OpenAI Realtime GA 协议结构与离线安全回归测试。"""

import asyncio
import json
import sys
from importlib import util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "ch15_agent" / "llm" / "18_openai_realtime_agent.py"
)
SPEC = util.spec_from_file_location("ch15_openai_realtime_agent", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_session_update_uses_current_nested_ga_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1-test")
    agent = MODULE.OpenAIRealtimeAgent(mock=True)

    event = json.loads(json.dumps(agent.session_update_event()))
    session = event["session"]
    assert event["type"] == "session.update"
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2.1-test"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm"}
    assert session["audio"]["output"]["voice"] == "marin"
    assert "modalities" not in session
    assert "input_audio_format" not in session
    assert "output_audio_format" not in session


def test_current_event_names_and_tool_call_shape_are_json_serializable():
    events = json.loads(json.dumps(MODULE.OpenAIRealtimeAgent.mock_server_events()))
    event_types = {event["type"] for event in events}
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio_transcript.delta" in event_types
    assert "response.audio.delta" not in event_types

    tool_event = MODULE.OpenAIRealtimeAgent._normalize(events[-1])
    assert tool_event == {
        "type": "tool_call",
        "name": "get_weather",
        "call_id": "call_mock_weather",
        "args": {"city": "北京"},
    }


def test_default_mock_never_opens_a_socket(monkeypatch):
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from websockets.asyncio import client as websocket_client

    def fail_connect(*args, **kwargs):
        raise AssertionError("mock 模式不应调用 WebSocket connect")

    monkeypatch.setattr(websocket_client, "connect", fail_connect)
    agent = MODULE.OpenAIRealtimeAgent()
    asyncio.run(agent.connect())

    assert agent.mock is True
    assert agent.ws is None


def test_real_connect_sends_non_null_session_without_beta_header(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "0")
    sent = []
    captured = {}

    class FakeWebSocket:
        async def send(self, payload):
            sent.append(json.loads(payload))

    async def fake_connect(url, *, additional_headers):
        captured["url"] = url
        captured["headers"] = additional_headers
        return FakeWebSocket()

    from websockets.asyncio import client as websocket_client

    monkeypatch.setattr(websocket_client, "connect", fake_connect)
    agent = MODULE.OpenAIRealtimeAgent(api_key="test-token", mock=False)
    asyncio.run(agent.connect())

    assert captured["url"].endswith("?model=gpt-realtime-2.1")
    assert captured["headers"] == {"Authorization": "Bearer test-token"}
    assert "OpenAI-Beta" not in captured["headers"]
    assert sent == [agent.session_update_event()]
    assert sent[0]["session"] is not None


def test_real_connect_requires_key_before_network(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = MODULE.OpenAIRealtimeAgent(mock=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(agent.connect())


def test_mock_false_cannot_bypass_unset_global_gate(monkeypatch):
    monkeypatch.delenv("LLM_MOCK", raising=False)
    agent = MODULE.OpenAIRealtimeAgent(api_key="must-not-be-used", mock=False)
    assert agent.mock is True
