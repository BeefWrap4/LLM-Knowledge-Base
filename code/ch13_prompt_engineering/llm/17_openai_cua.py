# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.3 OpenAI Computer-Using Agent (CUA)
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai
# run: python 17_openai_cua.py
# expected_runtime: 10-20s (real api)
# expected_output: 无 Key 时 [SKIP]；有 Key 时只打印待验证动作，不执行
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.3
# Interview hooks:
# - OpenAI CUA 与 Claude Computer Use 的协议层差异？
# - reasoning.effort 三档对成本和延迟的影响？
# - 宿主为何必须独立校验动作并在隔离环境执行？

import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _real_api_ready() -> bool:
    if os.environ.get("LLM_MOCK") != "0":
        print("[SKIP] 离线安全模式：只有显式设置 LLM_MOCK=0 才会调用 OpenAI API")
        print("OK")
        return False
    if OpenAI is None or not os.environ.get("OPENAI_API_KEY"):
        print("[SKIP] 真实调用需要 openai 和 OPENAI_API_KEY")
        print("OK")
        return False
    return True


def call_openai_computer(user_msg: str):
    if not _real_api_ready():
        return None
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        tools=[{"type": "computer"}],
        input=user_msg,
    )


if __name__ == "__main__":
    response = call_openai_computer(
        "在隔离浏览器中打开公司公开主页；不要登录、下载或提交表单。"
    )
    if response is None:
        raise SystemExit(0)

    # 只打印模型建议；真实宿主须逐项授权、执行，再按 call_id 回传截图。
    for item in response.output:
        if item.type == "computer_call":
            for action in item.actions:
                print("[待验证动作]", action)
    print("[安全默认] 本示例不执行任何宿主 GUI 动作")
    print("OK")
