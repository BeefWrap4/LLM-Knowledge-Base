# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.lcel_style_basic_chain
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain-core, langchain-openai
# run: python 01_lcel_style_basic_chain.py
# expected_runtime: <1s (mock mode)
# expected_output: joke string
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. 什么是 LCEL？它相比传统的 LLMChain 有哪些优势？
#   2. 请解释 `prompt | model | parser` 管道符的运行机制。
# LCEL 风格：声明式链式调用
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# Mock LLM (no real API call) for offline runnability
# 包装为 RunnableLambda 以兼容 LCEL 管道 (`|`)
def _mock_llm_invoke(msgs):
    return "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。"


model = RunnableLambda(_mock_llm_invoke)

prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
output_parser = StrOutputParser()

# 管道式组合：prompt | model | parser
chain = prompt | model | output_parser

# 一行调用
result = chain.invoke({"topic": "程序员"})
print(result)

if __name__ == "__main__":
    print("OK")
