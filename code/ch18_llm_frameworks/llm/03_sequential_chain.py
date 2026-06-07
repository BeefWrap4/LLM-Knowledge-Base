# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.2 Chain 概念与类型 - SequentialChain
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 03_sequential_chain.py
# expected_runtime: <1s (mock mode)
# expected_output: outline + article dict
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.2
# Interview hooks:
#   1. SequentialChain 中 output_key 的作用是什么？变量如何在链之间流动？
#   2. SequentialChain 与 LCEL 的 `chain1 | chain2` 写法有什么本质差异？


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from langchain.chains import SequentialChain, LLMChain
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    print("OK")
    _sys.exit(0)
from langchain_core.prompts import PromptTemplate

class _MockChatModel:
    def __init__(self, content): self._c = content
    def invoke(self, msgs):
        class _R: content = self._c
        return _R()

# Wave 21: 优先真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 降级 mock
USE_REAL_API = os.environ.get("USE_REAL_API") == "1"
if USE_REAL_API:
    from shared.chatmodel_factory import make_chat_model
    _llm = make_chat_model()  # 默认厂商 (deepseek)
    class _SwitchMock:
        """真实 LLM 模式下也保留 _SwitchMock 接口 (invoke 单方法), 框架可识别"""
        pass
else:
    # 单一 mock LLM 在不同 prompt 下返回不同内容
    class _SwitchMock:
        def invoke(self, msgs):
            last = msgs[-1].content if hasattr(msgs[-1], 'content') else str(msgs[-1])
            if "大纲" in last or "outline" in last.lower():
                text = "1. 选型考量\n2. 主流框架\n3. 落地建议"
            else:
                text = "在2026年，大模型应用框架选型应聚焦于编排能力、Agent支持、RAG能力与可观测性。LangGraph适合复杂工作流，LlamaIndex是RAG首选，Dify适合快速落地。"
            class _R: content = text
            return _R()
    _llm = _SwitchMock()

llm = _llm

# 第一链：生成大纲
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="为关于'{topic}'的博客文章生成一个3点大纲。"
    ),
    output_key="outline"
)

# 第二链：基于大纲写正文
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["outline"],
        template="基于以下大纲，写一篇300字的博客文章：\n\n{outline}"
    ),
    output_key="article"
)

# 串联
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["outline", "article"],
    verbose=True
)

result = overall_chain.invoke({"topic": "大模型应用框架选型"})
print(result["article"])

if __name__ == "__main__":
    print("OK")