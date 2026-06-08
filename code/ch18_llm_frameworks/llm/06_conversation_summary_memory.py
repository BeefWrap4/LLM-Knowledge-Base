# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.3 Memory 机制 - ConversationSummaryMemory
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 06_conversation_summary_memory.py
# expected_runtime: <1s (mock mode)
# expected_output: summarized history
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.3
# Interview hooks:
#   1. ConversationSummaryMemory 如何平衡上下文长度和信息保留？
#   2. max_token_limit 起什么作用？过小会丢失什么？


# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from langchain.memory import ConversationSummaryMemory

    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print(
    "OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)"
)
from langchain.chains import ConversationChain


class _MockChatModel:
    def invoke(self, msgs):
        joined = " ".join([m.content for m in msgs if hasattr(m, "content")])
        if "摘要" in joined or "Summarize" in joined or "请用一句话" in joined:
            text = "用户叫王五，在北京工作，是一名软件工程师，团队有10人。"
        elif "工作" in joined:
            text = "您是一名软件工程师。"
        else:
            text = "好的，已记录。"

        class _R:
            content = text

        return _R()


llm = _MockChatModel()
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True,
    max_token_limit=500,  # 摘要的最大 Token 数
)

conversation = ConversationChain(llm=llm, memory=memory)

# 多轮对话后，早期对话被压缩为摘要
conversation.predict(input="我叫王五，在北京工作。")
conversation.predict(input="我是一名软件工程师。")
conversation.predict(input="我的团队有10个人。")
conversation.predict(input="我在做什么工作？")  # ✓ 从摘要或历史中获取

print(memory.load_memory_variables({})["history"])
# System: 用户叫王五，在北京工作，是一名软件工程师，团队有10人。
# Human: 我在做什么工作？
# AI: 你是一名软件工程师。

if __name__ == "__main__":
    print("OK")
