# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.5.2 Gemini Explicit Caching
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: google-genai
# run: python 15_gemini_explicit_caching.py
# expected_runtime: 5-15s (real api)
# expected_output: 打印 cached tokens 与 prompt tokens
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.2
# Interview hooks:
# - Gemini 的"显式缓存"与 OpenAI 自动缓存差别？
# - Gemini 缓存跨 session 共享带来什么收益和风险？
# - 如何对长文档 RAG 场景设计最优缓存 TTL？

import os

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def _real_api_ready() -> bool:
    if os.environ.get("LLM_MOCK") != "0":
        print("[SKIP] 离线安全模式：只有显式设置 LLM_MOCK=0 才会调用 Gemini API")
        print("OK")
        return False
    if genai is None or types is None or not os.environ.get("GEMINI_API_KEY"):
        print("[SKIP] 真实调用需要 google-genai 和 GEMINI_API_KEY")
        print("OK")
        return False
    return True


def run_gemini_cache_demo():
    if not _real_api_ready():
        return None
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    large_handbook_doc = "公司年假政策……" * 5000
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    # 1. 显式创建缓存；省略 ttl 时默认 1 小时
    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name="company-handbook-cache",
            system_instruction="你是企业知识库助手。",
            contents=[large_handbook_doc],
            ttl="3600s",
        ),
    )

    # 2. 使用缓存进行推理
    response = client.models.generate_content(
        model=model_name,
        contents="公司年假政策是什么？",
        config=types.GenerateContentConfig(cached_content=cache.name),
    )
    return response


if __name__ == "__main__":
    response = run_gemini_cache_demo()
    if response is None:
        raise SystemExit(0)
    print(response.text)

    # 3. 查询缓存用量
    usage = response.usage_metadata
    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    print(f"缓存命中 tokens: {cached_tokens}")
    print(f"新输入 tokens:   {max(prompt_tokens - cached_tokens, 0)}")
    print("OK")
