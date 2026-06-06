# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.2 Chain 概念与类型 - LLMChain
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 02_llmchain_basic.py
# expected_runtime: <1s (mock mode)
# expected_output: ad copy string
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.2
# Interview hooks:
#   1. LLMChain 的 input_variables 有什么作用？模板插值的机制是什么？
#   2. 为什么 LLMChain 在 LCEL 出现后被认为"过时"？
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

class _MockChatModel:
    def invoke(self, msgs):
        class _R: content = "这款智能手表专为运动爱好者打造——实时心率、GPS轨迹、50米防水，让每一次奔跑都更专业。"
        return _R()

llm = _MockChatModel()

prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="为{product}写一段面向{audience}的广告文案，50字以内。"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"product": "智能手表", "audience": "运动爱好者"})
print(result["text"])

if __name__ == "__main__":
    print("OK")
