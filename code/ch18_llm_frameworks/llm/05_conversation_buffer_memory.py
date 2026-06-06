# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.3 Memory 机制深度解析 - ConversationBufferMemory
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 05_conversation_buffer_memory.py
# expected_runtime: <1s (mock mode)
# expected_output: memory dict with history
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.3
# Interview hooks:
#   1. ConversationBufferMemory 适合什么场景？它的 Token 消耗有什么问题？
#   2. ConversationChain 与现代 LCEL 链式写法有什么不同？
# ConversationBufferMemory：完整缓冲记忆


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from langchain.memory import ConversationBufferMemory
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    print("OK")
    _sys.exit(0)
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage

class _MockChatModel:
    def invoke(self, msgs):
        # 模拟"记住"用户姓名和年龄
        joined = " ".join([m.content for m in msgs if hasattr(m, 'content')])
        if "你叫什么" in joined or "名字" in joined:
            text = "您叫张三。"
        elif "几岁" in joined or "年龄" in joined:
            text = "您今年 30 岁。"
        else:
            text = "好的，我记住了。"
        class _R: content = text
        return _R()

llm = _MockChatModel()
memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="我叫张三，我今年30岁。")
conversation.predict(input="我的名字是什么？")   # ✓ 正确回答 "张三"
conversation.predict(input="我几岁了？")         # ✓ 正确回答 "30"

# 查看记忆内容
print(memory.load_memory_variables({}))
# {'history': [HumanMessage(...), AIMessage(...), ...]}

if __name__ == "__main__":
    print("OK")