# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.2 Anthropic Prompt Caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic (可选，缺失则使用 mock)
# run: python 13_anthropic_prompt_caching.py
# expected_runtime: <1s (mock) / 5-15s (real api)
# expected_output: 打印缓存创建/读取 token 数及命中率
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.2
# Interview hooks:
# - Anthropic 的 ephemeral (5min) 与 1h cache 各适合什么场景？
# - cache_control 应该插在 messages 的哪一段最有效？
# - 如何监控命中率并触发告警？

import os

USE_MOCK = os.environ.get("USE_REAL_API") != "1"


def load_user_document() -> str:
    """模拟加载一份 50K tokens 的文档。"""
    return "示例代码 - " * 1000


class _MockUsage:
    cache_creation_input_tokens = 8000
    cache_read_input_tokens = 32000
    input_tokens = 400


class _MockResp:
    usage = _MockUsage()


def call_anthropic_with_cache(system_prompt, user_content):
    if USE_MOCK:
        return _MockResp()
    import anthropic
    client = anthropic.Anthropic()
    return client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}]
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
            "cache_control": {"type": "ephemeral"}  # 5 分钟缓存
        }
    ]

    # 长文档（每次请求不同，但前缀可复用）
    long_document = load_user_document()  # 假设 50K tokens

    user_content = [
        {"type": "text", "text": f"<document>{long_document}</document>"},
        {"type": "text", "text": "请审查上述代码的安全漏洞。", "cache_control": {"type": "ephemeral"}}
    ]

    response = call_anthropic_with_cache(system_prompt, user_content)

    # 检查缓存命中情况
    print(f"缓存创建: {response.usage.cache_creation_input_tokens}")
    print(f"缓存读取: {response.usage.cache_read_input_tokens}")
    print(f"新输入:   {response.usage.input_tokens}")
    hit_rate = response.usage.cache_read_input_tokens / max(
        response.usage.cache_read_input_tokens + response.usage.input_tokens, 1
    )
    print(f"缓存命中率: {hit_rate:.2%}")
