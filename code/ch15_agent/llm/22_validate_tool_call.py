# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.10 高频题13 - 工具调用幻觉校验伪代码
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 22_validate_tool_call.py
# expected_runtime: <1s
# expected_output: 校验结果（白名单 / 必填参数 / 通过）
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.10-高频题13工具幻觉怎么办
# Interview hooks:
#   1. 模型编造工具名时怎么办？白名单机制
#   2. 模型漏传必填参数时怎么提示它自己修正？
#   3. Schema 校验是"安全护栏"还是"体验优化"？两者结合的实战示例？

# 伪代码（带可运行实现演示）
ALLOWED_TOOLS = {"get_weather", "search_web", "send_email"}
TOOL_SCHEMAS = {
    "get_weather": {"required": ["city"], "properties": {"city": {"type": "string"}}},
    "search_web": {"required": ["query"], "properties": {"query": {"type": "string"}}},
    "send_email": {
        "required": ["to", "subject", "body"],
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
}


def validate_tool_call(tool_name, args):
    if tool_name not in ALLOWED_TOOLS:
        return False, f"工具 '{tool_name}' 不存在"
    schema = TOOL_SCHEMAS[tool_name]
    for param in schema["required"]:
        if param not in args:
            return False, f"缺少必填参数 '{param}'"
    return True, "校验通过"


def main():
    cases = [
        ("get_weather", {"city": "北京"}),
        ("search_web", {"query": "python gil"}),
        ("send_email", {"to": "a@b.c", "subject": "hi", "body": "yes"}),
        ("send_email", {"to": "a@b.c"}),  # 缺 subject/body
        ("create_image", {"prompt": "a cat"}),  # 工具不在白名单
    ]
    for tool, args in cases:
        ok, reason = validate_tool_call(tool, args)
        print(f"  validate({tool}, {args}) -> ok={ok!s:5} | {reason}")
    print("\nOK")


if __name__ == "__main__":
    main()
