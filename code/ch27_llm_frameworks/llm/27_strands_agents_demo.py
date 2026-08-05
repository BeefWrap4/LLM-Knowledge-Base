# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.strands_agents_demo
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: strands-agents (mocked structure)
# run: python 27_strands_agents_demo.py
# expected_runtime: <1s
# expected_output: agent config dump
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. Strands 的 BidiAgent 与传统 stream 有什么区别？真正的"双向流"意味着什么？
#   2. Strands 与 Anthropic 官方推荐的 Claude Agent SDK 范式如何对应？
"""
Strands Agents SDK 实战：双向流式研究 Agent - 离线 mock 结构
"""
import os

# 真实环境:
# from strands import Agent, tool
# from strands.models import BedrockModel
# from strands.tools.mcp import MCPClient
# from mcp import stdio_client, StdioServerParameters


# ===== 1. 定义工具 =====
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 当前晴，25°C"


# ===== 2. MCP Client 配置（mock）=====
mcp_config = {
    "command": "uvx",
    "args": ["awslabs.aws-documentation-mcp-server"],
}

# ===== 3. Bedrock 模型配置 =====
model_config = {
    "model_id": os.environ.get("BEDROCK_MODEL_ID", "<set BEDROCK_MODEL_ID>"),
    "temperature": 0.7,
}

# ===== 4. Agent 配置（mock BidiAgent）=====
agent_config = {
    "model": model_config,
    "tools": [get_weather, *([])],  # 真实会拼接 mcp_client.list_tools_sync()
    "system_prompt": "你可以使用 AWS 文档 MCP 工具和天气工具回答问题。",
    "bidi_stream": True,  # 双向流：边思考边输出，边调工具
}

# ===== 5. 模拟 stream_async 事件 =====
print("=== Strands Agent 配置 ===")
for k, v in agent_config.items():
    print(f"  {k}: {str(v)[:80]}")
print(f"\nMCP Server: {mcp_config['command']} {mcp_config['args']}")
print("\n 双向流：边思考边输出，边调工具")

# 模拟事件流
print("\n=== 模拟流式事件 ===")
mock_events = [
    {"data": "正在比较"},
    {"data": " 已配置的 Bedrock 模型"},
    {"tool_use": {"name": "get_weather", "args": {"city": "Seattle"}}},
    {"data": " 在当前区域的能力与成本..."},
]
for ev in mock_events:
    if "data" in ev:
        print(ev["data"], end="", flush=True)
    elif "tool_use" in ev:
        print(f"\n[调用工具: {ev['tool_use']['name']}]")
print()

if __name__ == "__main__":
    print("OK")
