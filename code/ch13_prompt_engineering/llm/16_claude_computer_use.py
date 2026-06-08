# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.3 Claude Computer Use
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic
# run: python 16_claude_computer_use.py
# expected_runtime: 10-20s (real api)
# expected_output: 打印 Claude Computer Use 工具规范与返回
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.3
# Interview hooks:
# - Computer Use 与传统 RPA 的区别？
# - 为何需要"观察-思考-动作"闭环？失败恢复如何实现？
# - 高风险操作（支付、删除）的拦截策略？

import base64

import anthropic

_client = anthropic.Anthropic()


def call_claude_computer_use(screenshot_b64: str, user_msg: str):
    # Computer Use 工具定义
    tools = [
        {
            "name": "computer",
            "description": "控制计算机：截图、点击、键入、滚动",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"enum": ["screenshot", "left_click", "type", "key", "scroll", "wait"]},
                    "coordinate": {"type": "array", "items": {"type": "integer"}},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            },
        }
    ]

    # System Prompt 关键要素
    COMPUTER_USE_SYSTEM = """
你是一个计算机使用助手，可控制浏览器完成用户任务。

【行为准则】
1. 每次执行动作前先观察当前截图
2. 动作之间保持简短思考：<thinking>目标→动作→预期</thinking>
3. 失败时截图诊断，重新规划
4. 完成任务的最后一步必须调用 computer(action="done")
5. 高风险操作（支付、删除等）必须先和用户确认

【截图标注】
返回坐标时使用 0-1000 归一化坐标，0=左上，1000=右下。
"""

    return _client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=COMPUTER_USE_SYSTEM,
        tools=tools,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": screenshot_b64}},
                    {"type": "text", "text": user_msg},
                ],
            }
        ],
    )


if __name__ == "__main__":
    # 使用一个空白 1x1 PNG 作为占位截图
    fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
    response = call_claude_computer_use(fake_png, "在搜索框中输入 'Python tutorial' 然后点击搜索按钮")
    for block in response.content:
        print(f"[{block.type}]", getattr(block, "text", None) or getattr(block, "input", None))
