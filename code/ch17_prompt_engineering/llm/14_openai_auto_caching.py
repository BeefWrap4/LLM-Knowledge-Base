# ---
# chapter: 18
# topic: Context Engineering
# topic_id: prompt_engineering.openai_auto_caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai
# run: python 14_openai_auto_caching.py
# expected_runtime: 3-10s (real api)
# expected_output: 打印请求构造后返回的内容与缓存约束说明
# ---
# See: ../../../18_Context_Engineering.md
# Interview hooks:
# - 隐式缓存与 GPT-5.6 显式断点各适合什么流量模式？
# - 如果前缀只改了 1 个 token，缓存能命中吗？
# - 如何把动态内容尽量"后置"以最大化缓存命中？

import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

LARGE_SYSTEM_PROMPT = "你是一名专业的中文写作助手。规则：" * 500


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


def call_openai(input_items):
    if not _real_api_ready():
        return None
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        input=input_items,
    )


if __name__ == "__main__":
    document = "示例长文档片段……" * 100
    user_query = "请提炼以上文档要点"

    input_items = [
        {
            "role": "system",
            "content": LARGE_SYSTEM_PROMPT,  # > 1024 tokens，自动进入缓存候选
        },
        {
            "role": "user",
            "content": f"文档：{document}\n问题：{user_query}",  # 动态部分
        },
    ]

    response = call_openai(input_items)
    if response is None:
        raise SystemExit(0)
    print(f"[Response] {response.output_text}")
    details = getattr(response.usage, "input_tokens_details", None)
    print(f"[Cached tokens] {getattr(details, 'cached_tokens', 0)}")
    print(f"[Cache-write tokens] {getattr(details, 'cache_write_tokens', 0)}")
    print("\n[OpenAI 缓存关键约束]")
    print(" - GPT-5.6 支持隐式缓存，也支持显式断点")
    print(" - 写入按未缓存输入的 1.25x 计价；读取按模型 cached-input 价格计价")
    print(" - 必须用真实 usage 与复用次数计算收益，不能只看前缀字符数")
    print("OK")
