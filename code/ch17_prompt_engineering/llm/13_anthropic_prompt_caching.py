# ---
# chapter: 18
# topic: Context Engineering
# topic_id: prompt_engineering.anthropic_prompt_caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic
# run: python 13_anthropic_prompt_caching.py
# expected_runtime: 5-15s (real api)
# expected_output: 打印缓存创建/读取 token 数及命中率
# ---
# See: ../../../18_Context_Engineering.md
# Interview hooks:
# - Anthropic 的 ephemeral (5min) 与 1h cache 各适合什么场景？
# - cache_control 应该插在 messages 的哪一段最有效？
# - 如何监控命中率并触发告警？

import os

try:
    import anthropic
except ImportError:
    anthropic = None


def load_user_document() -> str:
    """模拟加载一份 50K tokens 的文档。"""
    return "示例代码 - " * 1000


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


def call_anthropic_with_cache(system_prompt, user_content):
    if not _real_api_ready():
        return None
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )


if __name__ == "__main__":
    kb_text = "你是一位资深 Python 后端工程师，擅长代码审查和性能优化。" * 20  # 模拟知识库
    # 在 system prompt 中标记 cache_control 断点
    system_prompt = [
        {
            "type": "text",
            "text": "你是一位资深 Python 后端工程师，擅长代码审查和性能优化。请严格按 JSON 格式输出审查结果。",
        },
        {
            "type": "text",
            "text": f"<company_kb>\n{kb_text}\n</company_kb>",  # 大段静态内容
            "cache_control": {"type": "ephemeral"},  # 5 分钟缓存
        },
    ]

    # 长文档（每次请求不同，但前缀可复用）
    long_document = load_user_document()  # 假设 50K tokens

    user_content = [
        {"type": "text", "text": f"<document>{long_document}</document>"},
        {
            "type": "text",
            "text": "请审查上述代码的安全漏洞。",
        },
    ]

    response = call_anthropic_with_cache(system_prompt, user_content)
    if response is None:
        raise SystemExit(0)

    # 检查缓存命中情况
    print(f"缓存创建: {response.usage.cache_creation_input_tokens}")
    print(f"缓存读取: {response.usage.cache_read_input_tokens}")
    print(f"新输入:   {response.usage.input_tokens}")
    reuse_rate = response.usage.cache_read_input_tokens / max(
        response.usage.cache_read_input_tokens
        + response.usage.cache_creation_input_tokens
        + response.usage.input_tokens,
        1,
    )
    print(f"缓存 token 复用率: {reuse_rate:.2%}")
    print("说明：5 分钟写入为基础输入价 1.25x，1 小时写入为 2x；命中/刷新为 0.1x。")
    print("OK")
