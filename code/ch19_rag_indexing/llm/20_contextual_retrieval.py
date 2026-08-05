# ---
# chapter: 21
# topic: 生产级 RAG 系统
# topic_id: rag_indexing.contextual_retrieval
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic (optional, mocked in demo)
# run: python 20_contextual_retrieval.py
# expected_runtime: <1s (mock mode)
# expected_output: chunk enriched with context prefix
# ---
# See: ../../../21_生产级RAG系统.md
# Interview hooks:
#   1. Contextual Retrieval 相比传统 chunking 能降低多少检索失败率？
#   2. 用哪个 LLM 来生成上下文最划算？成本如何控制？
#   3. 为什么不直接用大上下文模型代替 RAG（成本/延迟权衡）？

# Contextual Retrieval 实现（Anthropic 官方推荐写法，mock 演示版）
import os

DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def add_context_to_chunk(chunk: str, full_document: str, client=None) -> str:
    """为每个 chunk 注入 LLM 生成的上下文描述"""
    prompt = f"""以下是文档的一个片段，请用 50-100 字简要说明这个片段的上下文，
    使其脱离原文后仍能独立理解。包括：文档主题、关键实体、与上下文的关系。

    完整文档（前 2000 字摘要）：
    {full_document[:2000]}

    目标片段：
    {chunk}

    上下文描述（简洁，不要重复片段内容）："""

    real_mode = os.getenv("LLM_MOCK", "1") == "0"
    if real_mode:
        if client is None:
            raise RuntimeError("真实模式需要显式传入 Anthropic client")
        response = client.messages.create(
            model=DEFAULT_ANTHROPIC_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        context = response.content[0].text
    else:
        # 默认 mock：即使调用者误传 client，也不读取 Key 或联网。
        context = f"本文档主题: {full_document[:50]}"
    return f"[上下文：{context}]\n\n{chunk}"  # 合并后一起 Embedding


if __name__ == "__main__":
    doc = "公司人事手册第一章 总则。员工每年享有 15 天带薪年假。请假需提前申请。"
    chunk = "员工每年享有 15 天带薪年假。"
    enriched = add_context_to_chunk(chunk, doc, client=None)
    print("原 chunk:")
    print(f"  {chunk}")
    print("\n注入上下文后:")
    print(enriched)
    print("OK")
