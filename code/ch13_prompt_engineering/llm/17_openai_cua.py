# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.3 OpenAI Computer-Using Agent (CUA)
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai
# run: python 17_openai_cua.py
# expected_runtime: 10-20s (real api)
# expected_output: 打印 CUA 返回的动作序列
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.3
# Interview hooks:
# - OpenAI CUA 与 Claude Computer Use 的协议层差异？
# - reasoning.effort 三档对成本和延迟的影响？
# - environment 参数（browser/mac/windows）如何影响动作集？

import base64

from openai import OpenAI

_client = OpenAI()


def call_openai_cua(screenshot_b64: str, user_msg: str):
    # OpenAI CUA 通过 Responses API 使用
    return _client.responses.create(
        model="computer-use-preview",  # 专用 CUA 模型
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
                {"type": "input_text", "text": user_msg}
            ]
        }],
        tools=[{
            "type": "computer_use_preview",
            "display_width": 1440,
            "display_height": 900,
            "environment": "browser"  # browser | mac | windows | linux
        }],
        reasoning={
            "summary": "auto",  # 返回思考摘要
            "effort": "medium"  # low / medium / high
        },
        truncation="auto"
    )


if __name__ == "__main__":
    fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
    response = call_openai_cua(fake_png, "登录这个网站并下载年度报告")

    # 解析 CUA 返回的动作
    for item in response.output:
        if item.type == "computer_call":
            action = item.action
            coords = getattr(action, "coordinates", None)
            print(f"动作: {action.type}, 坐标: {coords if coords else 'N/A'}")
