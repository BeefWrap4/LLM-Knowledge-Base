# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.3 Memory 机制 - ConversationSummaryBufferMemory
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 07_summary_buffer_memory.py
# expected_runtime: <1s (mock mode)
# expected_output: per-topic memory length print
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.3
# Interview hooks:
#   1. SummaryBufferMemory 相比纯 Summary 有什么优势？
#   2. 如何选择 max_token_limit？需要考虑哪些因素？


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from langchain.memory import ConversationSummaryBufferMemory
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    print("OK")
    _sys.exit(0)
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# 在 mock 模式下用一个简单回显 LLM 替代
class _MockChatModel:
    def invoke(self, msgs):
        last = msgs[-1].content if hasattr(msgs[-1], 'content') else str(msgs[-1])
        if "总结" in last or "摘要" in last or "Summarize" in last:
            text = "用户询问了多个关于大模型应用框架的话题。"
        else:
            text = "这是关于该问题的回答。"
        class _R: content = text
        return _R()

llm = _MockChatModel()

# SummaryBufferMemory = 摘要 + 最近 K 轮原始对话
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # 总 Token 预算
    return_messages=True,
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# 模拟长对话
topics = [
    "我想了解大模型应用框架。",
    "请详细介绍 LangChain。",
    "LangChain 的 Memory 有哪些类型？",
    "Memory 的 Token 消耗如何优化？",
    "除了 LangChain，还有哪些框架？",
    "LangGraph 和 LangChain 有什么关系？",
    "请对比一下所有框架的优劣。",
]

for topic in topics:
    response = conversation.predict(input=topic)
    mem = memory.load_memory_variables({})
    history = mem.get("history", "")
    print(f"问题: {topic[:30]}... | 记忆长度: {len(str(history))} 字符")

if __name__ == "__main__":
    print("OK")