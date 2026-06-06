# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.1 A2A Client 简化实现
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [httpx]
# run: python 15_a2a_client.py
# expected_runtime: 离线 <1s（无网络）；真实调用依赖目标服务
# expected_output: A2A 客户端结构演示（SSE 解析、JSON-RPC 请求构造）
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.9.1-A2A-协议Agent-Cards-JSON-RPC-over-HTTPSSE
# Interview hooks:
#   1. A2A 协议中 Agent Card 的标准托管路径是什么？/.well-known/agent.json
#   2. SSE 与普通 HTTP 长连接的本质区别？(服务端主动推送、Content-Type: text/event-stream)
#   3. A2A 和 MCP 在"能力描述"上的关键差异？(Agent Card 描述完整 Skills，MCP tools/list 列出原子工具)
"""
A2A Client 简化实现
展示 JSON-RPC over HTTP + SSE 流式通信
"""
import json
import uuid
from typing import AsyncIterator


class A2AClient:
    """
    A2A 协议客户端

    核心能力：
    1. 拉取 Agent Card（发现能力）
    2. 发送任务（JSON-RPC over HTTP）
    3. 订阅流式更新（SSE）
    """

    def __init__(self, agent_url: str, auth_token: str = None, mock: bool = True):
        self.agent_url = agent_url.rstrip("/")
        self.auth_token = auth_token
        self._card = None
        self.mock = mock

    async def fetch_agent_card(self) -> dict:
        """从 .well-known 路径拉取 Agent 能力描述"""
        if self.mock:
            self._card = {
                "name": "WeatherAgent",
                "version": "1.0.0",
                "skills": [
                    {"id": "get_weather", "name": "Get Weather",
                     "description": "获取天气信息",
                     "examples": ["北京今天天气怎么样？"]},
                ],
            }
            return self._card
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed; install via `pip install httpx`")
        async with httpx.AsyncClient() as client:
            url = f"{self.agent_url}/.well-known/agent.json"
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            self._card = response.json()
            return self._card

    async def send_task(self, skill_id: str, message: str, session_id: str = None) -> dict:
        """通过 JSON-RPC 2.0 发送任务到远程 Agent"""
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "skill": skill_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "sessionId": session_id or self._new_session_id(),
            }
        }
        if self.mock:
            # 直接返回模拟响应
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "task_id": str(uuid.uuid4()),
                    "status": "completed",
                    "artifacts": [{"type": "text", "text": f"[Mock {skill_id}] 收到: {message}"}],
                },
            }
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            response = await client.post(
                f"{self.agent_url}/rpc",
                json=rpc_request,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()

    async def stream_task(self, skill_id: str, message: str) -> AsyncIterator[dict]:
        """通过 SSE 订阅流式输出（mock 模式生成模拟事件）"""
        if self.mock:
            for i, chunk in enumerate(["正在", "查询", "天气", "..."]):
                yield {"event": "delta", "text": chunk, "index": i}
            yield {"event": "completed", "text": f"[Mock {skill_id}] 处理完成"}
            return

        rpc_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/sendSubscribe",
            "params": {
                "skill": skill_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                }
            }
        }
        headers = {"Accept": "text/event-stream"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.agent_url}/rpc/stream",
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
    def _new_session_id() -> str:
        return str(uuid.uuid4())


async def main():
    """离线演示 A2A 客户端三个核心能力"""
    client = A2AClient("https://weather-agent.example.com", mock=True)

    card = await client.fetch_agent_card()
    print(f"Agent: {card['name']} v{card['version']}")
    print(f"Skills: {[s['id'] for s in card['skills']]}")

    result = await client.send_task("get_weather", "北京今天天气怎么样？")
    print(f"\nResult: {result['result']}")

    print("\nStream events:")
    async for event in client.stream_task("get_weather", "上海未来三天预报"):
        print(f"  {event}")
    print("\nOK")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
