import sys as _sys_path_setup
from pathlib import Path as _Path_setup
_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.2 OpenAI Automatic Caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai (可选，缺失则使用 mock)
# run: python 14_openai_auto_caching.py
# expected_runtime: <1s (mock) / 3-10s (real api)
# expected_output: 打印请求构造后返回的 mock content 与缓存约束说明
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.2
# Interview hooks:
# - OpenAI 自动缓存触发条件 (≥1024 tokens) 背后的设计考量？
# - 如果前缀只改了 1 个 token，缓存能命中吗？
# - 如何把动态内容尽量"后置"以最大化缓存命中？

import os

USE_MOCK = os.environ.get("USE_REAL_API") != "1"

LARGE_SYSTEM_PROMPT = (
    "你是一名专业的中文写作助手。规则：" * 200
)  # 模拟一个 > 1024 tokens 的 system prompt


class _MockChoice:
    class _Msg:
        content = "[mock] 这是基于自动缓存命中的响应"
    message = _Msg()


class _MockResp:
    choices = [_MockChoice()]


def call_openai(messages):
    if USE_MOCK:
        return _MockResp()
    # Wave 16: 改用 UnifiedClient (支持 deepseek/kimi/siliconflow/MiniMax)
    from shared.llm_client import UnifiedClient
    client = UnifiedClient()
    return client.chat(
        messages=messages,  # 不传 model = 用 provider 默认 (deepseek-chat, MiniMax-Text-01, etc.)
    )


if __name__ == "__main__":
    document = "示例长文档片段……" * 100
    user_query = "请提炼以上文档要点"

    messages = [
        {
            "role": "system",
            "content": LARGE_SYSTEM_PROMPT  # > 1024 tokens，自动进入缓存候选
        },
        {
            "role": "user",
            "content": f"文档：{document}\n问题：{user_query}"  # 动态部分
        }
    ]

    response = call_openai(messages)
    # Wave 24: 适配 UnifiedClient 的 _LLMResponse (无 .choices 属性)
    content = response.content if hasattr(response, "content") else response.choices[0].message.content
    print(f"[Response] {content}")
    print("\n[OpenAI 缓存关键约束]")
    print(" - 缓存窗口：默认 5-10 分钟（动态 TTL）")
    print(" - 最小缓存前缀：1024 tokens")
    print(" - 仅在 Prompt 前缀严格相同时才命中")
