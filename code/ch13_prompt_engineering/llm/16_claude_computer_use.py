# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.3 Claude Computer Use
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic
# run: python 16_claude_computer_use.py
# expected_runtime: 10-20s (real api)
# expected_output: 无 Key 时 [SKIP]；有 Key 时只打印待验证动作，不执行
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.3
# Interview hooks:
# - Computer Use 与传统 RPA 的区别？
# - 为何需要"观察-思考-动作"闭环？失败恢复如何实现？
# - 高风险操作（支付、删除）的拦截策略？

import os

try:
    import anthropic
except ImportError:
    anthropic = None


def _real_api_ready() -> bool:
    if os.environ.get("LLM_MOCK") != "0":
        print("[SKIP] 离线安全模式：只有显式设置 LLM_MOCK=0 才会调用 Anthropic API")
        print("OK")
        return False
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        print("[SKIP] 真实调用需要 anthropic 和 ANTHROPIC_API_KEY")
        print("OK")
        return False
    return True


def call_claude_computer_use(user_msg: str):
    if not _real_api_ready():
        return None
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tools = [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
            "display_number": 1,
        }
    ]
    return client.beta.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
        max_tokens=2048,
        tools=tools,
        betas=["computer-use-2025-11-24"],
        messages=[{"role": "user", "content": user_msg}],
    )


if __name__ == "__main__":
    response = call_claude_computer_use(
        "在隔离浏览器中搜索 Python 官方 tutorial；不要登录、下载或提交表单。"
    )
    if response is None:
        raise SystemExit(0)
    for block in response.content:
        if block.type == "tool_use" and block.name == "computer":
            print("[待验证动作]", block.input)
        elif block.type == "text":
            print("[文本]", block.text)
    print("[安全默认] 本示例不执行任何宿主 GUI 动作")
    print("OK")
