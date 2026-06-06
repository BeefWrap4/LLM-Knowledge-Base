# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.8.2 防线2：工具调用幻觉
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 11_tool_hallucination_guard.py
# expected_runtime: <1s
# expected_output: 白名单/Schema 校验结果
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.8.2-Agent-工程化安全五道防线
# Interview hooks:
#   1. 工具调用幻觉常见的三种模式？(虚构工具名、参数缺失、参数类型错)
#   2. 为什么 Schema 校验要在执行前而不是执行后？(防止破坏性副作用)
#   3. 校验失败时如何让模型"知错能改"？重试 prompt 怎么写？

class ToolHallucinationGuard:
    """工具调用幻觉防护"""

    def __init__(self, allowed_tools: set[str], schema_registry: dict):
        self.allowed_tools = allowed_tools
        self.schema_registry = schema_registry

    def validate(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """
        严格校验工具调用

        1. 工具名白名单校验
        2. 参数 Schema 校验
        3. 必填参数检查
        """
        # 白名单校验
        if tool_name not in self.allowed_tools:
            return False, f"工具 '{tool_name}' 不在白名单中"

        schema = self.schema_registry.get(tool_name, {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # 必填参数检查
        for param in required:
            if param not in arguments:
                return False, f"缺少必填参数 '{param}'"

        # 参数类型检查
        for key, value in arguments.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._type_check(value, expected_type):
                    return False, f"参数 '{key}' 类型错误，期望 {expected_type}"

        return True, "校验通过"

    @staticmethod
    def _type_check(value, expected_type: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True


def main():
    allowed = {"get_weather", "send_email"}
    schemas = {
        "get_weather": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
        },
        "send_email": {
            "type": "object",
            "required": ["to", "subject", "body"],
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        },
    }
    guard = ToolHallucinationGuard(allowed, schemas)

    cases = [
        ("get_weather", {"city": "北京"}, "正常"),
        ("search_baidu", {"q": "python"}, "工具不在白名单"),
        ("get_weather", {"city": 12345}, "参数类型错（应字符串）"),
        ("get_weather", {}, "缺少必填 city"),
        ("send_email", {"to": "a@b.c", "subject": "hi", "body": "yes"}, "正常"),
        ("send_email", {"to": "a@b.c"}, "缺少 subject/body"),
    ]

    for tool_name, args, desc in cases:
        ok, reason = guard.validate(tool_name, args)
        print(f"  [{desc:>20}] ok={ok!s:5} | {reason}")
    print("\nOK")


if __name__ == "__main__":
    main()
