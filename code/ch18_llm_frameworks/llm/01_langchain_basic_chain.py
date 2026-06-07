# ---
# chapter: 18
# topic: LangChain 基础 Chain (LLM + Prompt + OutputParser)
# section: 18.1
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: export OPENAI_API_KEY=sk-... && python 01_langchain_basic_chain.py
# expected_runtime: 5-30s (real API)
# ---
#
# See: ../tutorial/Ch18_LLM工程框架实战.md §18.1
# Cross-refs:
#   - Ch15.3 Function Calling (LangChain 底层)
#   - Ch13 Prompt Engineering (template 设计)
#
# Interview hooks:
#   - "LangChain Chain 是什么?"  →  prompt | llm | output_parser 三段式
#   - "为什么用 LCEL?"            →  声明式 pipe 语法, 异步/流式/批处理开箱即用
#   - "LangChain vs LCEL 区别?"  →  旧 Chain 类 vs 新 Runnable

# 真实 API 模式
def run_real():
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 Python 专家, 回答简洁。"),
        ("user", "{question}"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()  # LCEL pipe 语法

    result = chain.invoke({"question": "什么是 GIL? 一句话回答。"})
    print(f"Real API answer: {result}")


# Mock 模式 (无需 API key)
def run_mock():
    """用 langchain FakeListChatModel 离线验证 Chain 逻辑 (无需 shared.mock_llm)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    # 模拟 LLM: 固定返回 "GIL 是全局解释器锁。"
    fake_llm = FakeListChatModel(responses=["GIL 是全局解释器锁, 保证同一时刻只有一个线程执行 Python 字节码。"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 Python 专家。"),
        ("user", "{question}"),
    ])

    chain = prompt | fake_llm | StrOutputParser()
    result = chain.invoke({"question": "什么是 GIL?"})
    print(f"Mock answer: {result}")


if __name__ == "__main__":
    import os
    if os.environ.get("OPENAI_API_KEY"):
        print("=== Real API 模式 ===")
        run_real()
    else:
        print("=== Mock 模式 (无 OPENAI_API_KEY) ===")
        print("设置 OPENAI_API_KEY 切换到真实 API 模式\n")
        run_mock()
    print("\nOK")
