# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.2 Gemini Explicit Caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: google-generativeai (可选，缺失则使用 mock)
# run: python 15_gemini_explicit_caching.py
# expected_runtime: <1s (mock) / 5-15s (real api)
# expected_output: 打印 cached tokens 与 prompt tokens
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.2
# Interview hooks:
# - Gemini 的"显式缓存"与 OpenAI 自动缓存差别？
# - Gemini 缓存跨 session 共享带来什么收益和风险？
# - 如何对长文档 RAG 场景设计最优缓存 TTL？

import os

USE_MOCK = os.environ.get("USE_REAL_API") != "1"


class _MockUsageMeta:
    cached_content_token_count = 50000
    prompt_token_count = 50300


class _MockResponse:
    text = "[mock] 公司年假政策：员工每年享有 15 天带薪年假，按入职年限递增。"
    usage_metadata = _MockUsageMeta()


def run_gemini_cache_demo():
    if USE_MOCK:
        return _MockResponse()

    # Google Gemini Explicit Caching
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY"))

    large_handbook_doc = "公司年假政策……" * 5000

    # 1. 显式创建缓存（最长 60 分钟）
    cached_content = genai.caching.CachedContent.create(
        model="gemini-2.5-pro",
        display_name="company-handbook-cache",
        system_instruction="你是企业知识库助手。",
        contents=[large_handbook_doc],  # 长文档列表
        ttl="3600s"  # 1 小时 TTL
    )

    # 2. 使用缓存进行推理
    model = genai.GenerativeModel.from_cached_content(cached_content)
    response = model.generate_content("公司年假政策是什么？")
    return response


if __name__ == "__main__":
    response = run_gemini_cache_demo()
    print(response.text)

    # 3. 查询缓存用量
    usage = response.usage_metadata
    print(f"缓存命中 tokens: {usage.cached_content_token_count}")
    print(f"新输入 tokens:   {usage.prompt_token_count - usage.cached_content_token_count}")
