# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.4.5 MCP Server 简单实现
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 04_mcp_server_simplified.py
# expected_runtime: 极短（需要 stdin 输入，可用 echo '...' | python 04_mcp_server_simplified.py 测试）
# expected_output: JSON-RPC 2.0 响应（initialize / tools/list / tools/call）
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.4.5-MCP-Server-简单实现
# Interview hooks:
#   1. MCP 协议的三种传输层是什么？(stdio / SSE / HTTP Streamable)
#   2. 一次完整的 MCP 会话要经历哪四个握手步骤？
#   3. MCP Server 暴露 Tools/Resources/Prompts 三类能力，分别对应什么使用场景？
"""
MCP Server 简化实现示例
展示 MCP 协议的核心交互模式
"""
import json
import sys
from typing import Any

class MCPServer:
    """
    MCP Server 简化实现

    通信方式：stdio（标准输入输出上的 JSON-RPC）
    """

    def __init__(self):
        self.tools = {
            "read_file": self.read_file,
            "list_directory": self.list_directory,
        }

    def send(self, message: dict):
        """发送 JSON-RPC 消息"""
        print(json.dumps(message), flush=True)

    def recv(self) -> dict | None:
        """接收 JSON-RPC 消息"""
        try:
            line = sys.stdin.readline()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def handle_initialize(self, request_id: Any, params: dict) -> dict:
        """处理初始化请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "simple-filesystem-server",
                    "version": "1.0.0"
                }
            }
        }

    def handle_tools_list(self, request_id: Any) -> dict:
        """处理工具列表请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "读取文件内容",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "list_directory",
                        "description": "列出目录内容",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                ]
            }
        }

    def handle_tools_call(self, request_id: Any, params: dict) -> dict:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](**arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": result}],
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"错误: {str(e)}"}],
                        "isError": True
                    }
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"工具 {tool_name} 不存在"}
            }

    def read_file(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def list_directory(self, path: str) -> str:
        import os
        entries = os.listdir(path)
        return "\n".join(entries)

    def run(self):
        """主事件循环"""
        while True:
            request = self.recv()
            if request is None:
                break

            method = request.get("method", "")
            request_id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                response = self.handle_initialize(request_id, params)
            elif method == "tools/list":
                response = self.handle_tools_list(request_id)
            elif method == "tools/call":
                response = self.handle_tools_call(request_id, params)
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"未知方法: {method}"}
                }

            self.send(response)


def demo_json_rpc_responses():
    """离线演示 JSON-RPC 响应结构（不进入 stdio 事件循环）"""
    server = MCPServer()
    init_resp = server.handle_initialize(request_id=1, params={})
    list_resp = server.handle_tools_list(request_id=2)
    call_resp = server.handle_tools_call(
        request_id=3,
        params={"name": "nonexistent", "arguments": {}},
    )

    print("[initialize]", json.dumps(init_resp, ensure_ascii=False))
    print("[tools/list tools count]", len(list_resp["result"]["tools"]))
    print("[tools/call unknown]", json.dumps(call_resp, ensure_ascii=False))
    print("OK")


if __name__ == "__main__":
    # 默认演示 JSON-RPC 响应结构；如需 stdio 模式请取消下一行注释
    # server = MCPServer(); server.run()
    demo_json_rpc_responses()
