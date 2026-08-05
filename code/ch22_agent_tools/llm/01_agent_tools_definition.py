# ---
# chapter: 22
# topic: Agent 基础与工具调用
# topic_id: agent_tools.agent_tools_definition
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 01_agent_tools_definition.py
# expected_runtime: <1s
# expected_output: OK
# ---
# See: ../../../22_Agent基础与工具调用.md
# Interview hooks:
#   1. Function Calling 的 tool schema 必须包含哪些字段才能让模型正确选择？
#   2. tools 列表是放在 system message 还是 user message 里的？
#   3. 如何让模型知道"不该调用任何工具"？

# Agent 工具定义示例
TOOLS = [
    {
        "name": "web_search",
        "description": "搜索引擎，用于获取最新信息",
        "parameters": {"query": {"type": "string", "description": "搜索关键词"}},
    },
    {
        "name": "calculator",
        "description": "计算器，执行数学运算",
        "parameters": {"expression": {"type": "string", "description": "数学表达式"}},
    },
    {
        "name": "database_query",
        "description": "数据库查询",
        "parameters": {"sql": {"type": "string", "description": "SQL 语句"}},
    },
    {
        "name": "send_email",
        "description": "发送邮件",
        "parameters": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
]


def main():
    """演示工具 schema 的关键字段：name、description、parameters（properties + type + required）"""
    for tool in TOOLS:
        assert "name" in tool and "description" in tool and "parameters" in tool
        print(f"[OK] tool={tool['name']:>14} | params={list(tool['parameters'].keys())}")


if __name__ == "__main__":
    main()
