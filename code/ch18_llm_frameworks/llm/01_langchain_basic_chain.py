# ---
# chapter: 18
# topic: LangChain 基础 Chain (LLM + Prompt + OutputParser)
# section: 18.1
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: LLM_MOCK=1 python 01_langchain_basic_chain.py
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
#
# Note: 默认离线；只有显式 LLM_MOCK=0 且配置 OPENAI_API_KEY 才进入真实 API 路径。

import os


# 真实 API 模式
def main():
    if os.environ.get("LLM_MOCK") != "0":
        print("[SKIP] 离线模式：设置 LLM_MOCK=0 并配置 OPENAI_API_KEY 才会调用真实 API")
        print("LCEL: prompt | llm | output_parser")
        print("Mock answer: GIL 是 CPython 中限制同一时刻仅一个线程执行 Python 字节码的互斥锁。")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("[SKIP] 真实调用需要 OPENAI_API_KEY")
        return

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个 Python 专家, 回答简洁。"),
            ("user", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-5.6"))
    chain = prompt | llm | StrOutputParser()  # LCEL pipe 语法

    result = chain.invoke({"question": "什么是 GIL? 一句话回答。"})
    print(f"Real API answer: {result}")


if __name__ == "__main__":
    main()
    print("\nOK")
