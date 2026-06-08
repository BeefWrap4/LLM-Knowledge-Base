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
#
# Note: W3 之后, mock 实现已下沉到 tests/_mocks/demo_langchain_basic_chain.py
#       (LLM_MOCK=1 模式), 主流程仅保留真实 API 路径.

# 真实 API 模式
def main():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个 Python 专家, 回答简洁。"),
            ("user", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()  # LCEL pipe 语法

    result = chain.invoke({"question": "什么是 GIL? 一句话回答。"})
    print(f"Real API answer: {result}")


if __name__ == "__main__":
    main()
    print("\nOK")
