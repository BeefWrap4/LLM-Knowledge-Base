import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

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
    _sys.exit(0)
print(
    "OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)"
)
from langchain.chains import ConversationChain

# Wave 30+: 真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 时 raise
from shared.chatmodel_factory import make_chat_model

llm = make_chat_model()  # 默认厂商

memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

conversation.predict(input="我叫张三，我今年30岁。")
conversation.predict(input="我的名字是什么？")  # ✓ 正确回答 "张三"
conversation.predict(input="我几岁了？")  # ✓ 正确回答 "30"

# 查看记忆内容
print(memory.load_memory_variables({}))
# {'history': [HumanMessage(...), AIMessage(...), ...]}

if __name__ == "__main__":
    print("OK")
