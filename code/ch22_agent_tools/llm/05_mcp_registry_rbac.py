# ---
# chapter: 23
# topic: MCP、A2A 与 Skills 协议生态
# topic_id: agent_tools.mcp_registry_rbac
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 05_mcp_registry_rbac.py
# expected_runtime: <1s
# expected_output: 权限检查结果、可用 Server 列表、审计日志条目数
# ---
# See: ../../../23_MCP_A2A与Skills协议生态.md
# Interview hooks:
#   1. 当 MCP Server 数量从 5 增长到 100 时，工程化最该解决什么？
#   2. RBAC 工具级权限和角色级权限有什么区别？合并时怎么取最大权限？
#   3. 审计日志为什么要做敏感字段脱敏？(password/token/secret)
"""
MCP 工程化管理 - Server 注册中心 + 动态加载 + 权限控制
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PermissionLevel(Enum):
    """工具权限级别"""

    DENY = 0  # 禁止访问
    READ = 1  # 只读访问
    WRITE = 2  # 读写访问
    ADMIN = 3  # 完全控制


@dataclass
class MCPServerInfo:
    """MCP Server 注册信息"""

    name: str
    version: str
    transport: str  # "stdio" | "sse"
    endpoint: str  # 路径或 URL
    tools: list[dict] = field(default_factory=list)
    permissions: dict[str, PermissionLevel] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    health_status: str = "unknown"  # "healthy" | "unhealthy" | "unknown"
    metadata: dict = field(default_factory=dict)


class MCPRegistry:
    """
    MCP Server 注册中心

    功能：
    1. Server 注册与发现
    2. 健康检查
    3. 版本管理
    4. 工具级权限控制 (RBAC)
    """

    def __init__(self):
        self._servers: dict[str, MCPServerInfo] = {}
        self._user_roles: dict[str, list[str]] = {}  # user_id -> [role, ...]
        self._role_permissions: dict[str, dict[str, PermissionLevel]] = {}
        self._audit_log: list[dict] = []
        self._max_log_entries = 10000

    def register(self, server_info: MCPServerInfo) -> bool:
        """注册 MCP Server"""
        server_id = f"{server_info.name}@{server_info.version}"
        server_info.last_heartbeat = time.time()
        server_info.health_status = "healthy"
        self._servers[server_id] = server_info
        return True

    def discover(self, user_id: str) -> list[MCPServerInfo]:
        """
        为用户发现可用的 Server（根据权限过滤）

        Args:
            user_id: 用户ID

        Returns:
            用户有权限访问的 Server 列表
        """
        available = []
        user_perms = self._get_user_permissions(user_id)

        for server_id, server in self._servers.items():
            if server.health_status != "healthy":
                continue
            # 过滤用户有权限的工具
            allowed_tools = []
            for tool in server.tools:
                tool_name = tool.get("name", "")
                perm = user_perms.get(tool_name, PermissionLevel.DENY)
                if perm.value >= PermissionLevel.READ.value:
                    allowed_tools.append(tool)

            if allowed_tools:
                filtered_server = MCPServerInfo(
                    name=server.name,
                    version=server.version,
                    transport=server.transport,
                    endpoint=server.endpoint,
                    tools=allowed_tools,
                    permissions={
                        k: v for k, v in user_perms.items() if v.value >= PermissionLevel.READ.value
                    },
                    health_status=server.health_status,
                    metadata=server.metadata,
                )
                available.append(filtered_server)

        return available

    def check_permission(
        self, user_id: str, tool_name: str, required_level: PermissionLevel = PermissionLevel.READ
    ) -> bool:
        """检查用户是否有权限调用指定工具"""
        user_perms = self._get_user_permissions(user_id)
        actual = user_perms.get(tool_name, PermissionLevel.DENY)
        return actual.value >= required_level.value

    def _get_user_permissions(self, user_id: str) -> dict[str, PermissionLevel]:
        """获取用户的所有工具权限"""
        roles = self._user_roles.get(user_id, ["default"])
        merged: dict[str, PermissionLevel] = {}

        for role in roles:
            role_perms = self._role_permissions.get(role, {})
            for tool, perm in role_perms.items():
                if tool not in merged or perm.value > merged[tool].value:
                    merged[tool] = perm

        return merged

    def log_tool_call(
        self,
        user_id: str,
        tool_name: str,
        arguments: dict,
        result: str,
        duration_ms: float,
        success: bool,
    ):
        """记录工具调用审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self._hash_id(user_id),
            "tool_name": tool_name,
            "arguments": self._sanitize_args(arguments),
            "result_preview": result[:200] if result else "",
            "duration_ms": duration_ms,
            "success": success,
        }
        self._audit_log.append(entry)

        # 防止日志无限增长
        if len(self._audit_log) > self._max_log_entries:
            self._audit_log = self._audit_log[-self._max_log_entries // 2 :]

    def health_check(self) -> dict[str, str]:
        """对所有 Server 执行健康检查"""
        now = time.time()
        for server_id, server in self._servers.items():
            if now - server.last_heartbeat > 60:  # 60秒无心跳视为不健康
                server.health_status = "unhealthy"
        return {sid: s.health_status for sid, s in self._servers.items()}

    @staticmethod
    def _hash_id(user_id: str) -> str:
        """对用户ID做哈希处理（隐私保护）"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    @staticmethod
    def _sanitize_args(args: dict) -> dict:
        """清理敏感参数（如密码、token）"""
        sensitive_keys = {"password", "token", "secret", "api_key", "auth"}
        sanitized = {}
        for k, v in args.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = v
        return sanitized


# ============ 使用示例 ============


def demo_mcp_registry():
    """MCP Registry 使用演示"""
    registry = MCPRegistry()

    # 1. 定义角色权限：客服人员只能查询，管理员可以修改
    registry._role_permissions["customer_service"] = {
        "query_order": PermissionLevel.READ,
        "query_policy": PermissionLevel.READ,
        "search_knowledge": PermissionLevel.READ,
    }
    registry._role_permissions["admin"] = {
        "query_order": PermissionLevel.WRITE,
        "update_order": PermissionLevel.WRITE,
        "refund_request": PermissionLevel.ADMIN,
    }

    # 2. 分配角色
    registry._user_roles["alice"] = ["customer_service"]
    registry._user_roles["bob"] = ["admin"]

    # 3. 注册 Server
    registry.register(
        MCPServerInfo(
            name="order-system",
            version="1.2.0",
            transport="stdio",
            endpoint="/servers/order-mcp",
            tools=[
                {"name": "query_order", "description": "查询订单"},
                {"name": "update_order", "description": "更新订单"},
            ],
        )
    )

    # 4. 权限检查
    print(f"Alice 能否 query_order: {registry.check_permission('alice', 'query_order')}")
    print(f"Alice 能否 update_order: {registry.check_permission('alice', 'update_order')}")
    print(f"Bob 能否 update_order: {registry.check_permission('bob', 'update_order')}")

    # 5. 发现可用 Server
    alice_servers = registry.discover("alice")
    print(f"\nAlice 可用的 Server: {[s.name for s in alice_servers]}")
    print(f"Alice 可用的工具: {[t['name'] for s in alice_servers for t in s.tools]}")

    # 6. 记录审计日志（含敏感字段脱敏）
    registry.log_tool_call(
        user_id="alice",
        tool_name="query_order",
        arguments={"order_id": "#12345", "auth_token": "secret-xxx"},
        result="订单状态：已发货",
        duration_ms=45.2,
        success=True,
    )
    print(f"\n审计日志条目数: {len(registry._audit_log)}")
    print(f"脱敏后参数: {registry._audit_log[-1]['arguments']}")


if __name__ == "__main__":
    demo_mcp_registry()
    print("OK")
