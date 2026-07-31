# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.1 A2A Client 简化实现
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [httpx]
# run: python 15_a2a_client.py
# expected_runtime: 离线 <1s（无网络）；真实调用依赖目标服务
# expected_output: A2A v1.0 客户端结构演示（Agent Card、JSON-RPC、SSE）
# ---
# See: ../tutorial/15_Agent智能体开发.md § 15.9.1
# Interview hooks:
#   1. A2A v1.0 Agent Card 的标准托管路径是什么？/.well-known/agent-card.json
#   2. SSE 与普通 HTTP 长连接的本质区别？(服务端主动推送、Content-Type: text/event-stream)
#   3. A2A 和 MCP 在"能力描述"上的关键差异？(Agent Card 描述完整 Skills，MCP tools/list 列出原子工具)
"""
A2A v1.0 Client 简化实现。

按 v1.0 使用 Agent Card 的 supportedInterfaces 发现 JSON-RPC 端点，
并演示 SendMessage / SendStreamingMessage。省略签名校验、重试与完整错误映射。
"""

import json
import uuid
from collections.abc import AsyncIterator


class A2AClient:
    """
    A2A 协议客户端

    核心能力：
    1. 拉取 Agent Card（发现能力）
    2. 发送任务（JSON-RPC over HTTP）
    3. 订阅流式更新（SSE）
    """

    def __init__(self, agent_url: str, auth_token: str | None = None, mock: bool = True):
        self.agent_url = agent_url.rstrip("/")
        self.auth_token = auth_token
        self._card = None
        self._rpc_url: str | None = None
        self.mock = mock

    async def fetch_agent_card(self) -> dict:
        """从 v1.0 well-known 路径拉取 Agent Card 并选择 JSON-RPC 绑定。"""
        if self.mock:
            self._card = {
                "name": "WeatherAgent",
                "version": "1.0.0",
                "description": "查询天气信息",
                "supportedInterfaces": [
                    {
                        "url": f"{self.agent_url}/a2a",
                        "protocolBinding": "JSONRPC",
                        "protocolVersion": "1.0",
                    }
                ],
                "capabilities": {"streaming": True},
                "securitySchemes": {
                    "bearer": {
                        "httpAuthSecurityScheme": {
                            "scheme": "Bearer",
                            "bearerFormat": "JWT",
                        }
                    }
                },
                "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
                "defaultInputModes": ["text/plain"],
                "defaultOutputModes": ["text/plain"],
                "skills": [
                    {
                        "id": "get_weather",
                        "name": "Get Weather",
                        "description": "获取天气信息",
                        "tags": ["weather", "forecast"],
                        "examples": ["北京今天天气怎么样？"],
                        "inputModes": ["text/plain"],
                        "outputModes": ["text/plain"],
                    },
                ],
            }
            self._rpc_url = self._card["supportedInterfaces"][0]["url"]
            return self._card
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed; install via `pip install httpx`")
        async with httpx.AsyncClient() as client:
            url = f"{self.agent_url}/.well-known/agent-card.json"
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            self._card = response.json()
            self._rpc_url = next(
                item["url"]
                for item in self._card["supportedInterfaces"]
                if item["protocolBinding"] == "JSONRPC" and item["protocolVersion"] == "1.0"
            )
            return self._card

    async def send_message(
        self,
        text: str,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> dict:
        """通过 JSON-RPC 2.0 的 SendMessage 启动或继续 A2A task。"""
        message = {
            "messageId": self._new_id(),
            "role": "ROLE_USER",
            "parts": [{"text": text}],
        }
        if task_id:
            message["taskId"] = task_id
        if context_id:
            message["contextId"] = context_id
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {"message": message},
        }
        if self.mock:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "task": {
                        "id": self._new_id(),
                        "contextId": context_id or self._new_id(),
                        "status": {"state": "TASK_STATE_COMPLETED"},
                        "artifacts": [
                            {
                                "artifactId": self._new_id(),
                                "parts": [{"text": f"[Mock] 收到: {text}"}],
                            }
                        ],
                    }
                },
            }
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")
        if self._rpc_url is None:
            raise RuntimeError("请先调用 fetch_agent_card() 发现 JSON-RPC 端点")
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "A2A-Version": "1.0",
            }
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            response = await client.post(
                self._rpc_url,
                json=rpc_request,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()

    async def stream_message(self, text: str) -> AsyncIterator[dict]:
        """通过 SendStreamingMessage + SSE 订阅流式输出。"""
        if self.mock:
            for i, chunk in enumerate(["正在", "查询", "天气", "..."]):
                yield {"event": "delta", "text": chunk, "index": i}
            yield {"event": "completed", "text": f"[Mock] 已处理: {text}"}
            return

        rpc_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "SendStreamingMessage",
            "params": {
                "message": {
                    "messageId": self._new_id(),
                    "role": "ROLE_USER",
                    "parts": [{"text": text}],
                },
            },
        }
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "A2A-Version": "1.0",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")
        if self._rpc_url is None:
            raise RuntimeError("请先调用 fetch_agent_card() 发现 JSON-RPC 端点")

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self._rpc_url,
                json=rpc_request,
                headers=headers,
                timeout=None,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data and data != "[DONE]":
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                continue

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def unwrap_send_message_response(response: dict) -> tuple[str, dict]:
        """提取 SendMessageResponse 的 task/message oneof。"""
        payload = response.get("result", {})
        variants = [name for name in ("task", "message") if name in payload]
        if len(variants) != 1:
            raise ValueError("SendMessageResponse.result 必须且只能包含 task 或 message")
        variant = variants[0]
        return variant, payload[variant]


async def main():
    """离线演示 A2A 客户端三个核心能力"""
    client = A2AClient("https://weather-agent.example.com", mock=True)

    card = await client.fetch_agent_card()
    print(f"Agent: {card['name']} v{card['version']}")
    print(f"Skills: {[s['id'] for s in card['skills']]}")
    print(f"JSON-RPC endpoint: {client._rpc_url}")

    result = await client.send_message("北京今天天气怎么样？")
    response_type, payload = client.unwrap_send_message_response(result)
    print(f"\n{response_type.title()} result: {payload}")

    print("\nStream events:")
    async for event in client.stream_message("上海未来三天预报"):
        print(f"  {event}")
    print("\nOK")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
