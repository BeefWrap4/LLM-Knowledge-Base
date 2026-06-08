"""
LangChain Chain 离线 demo (用 FakeListChatModel, 仅 CI/教学用).

W3 基建后, 主流程 (ch18/01) 已不再依赖 mock. 本文件保留作为
CI 教学 demo, 在 LLM_MOCK=1 模式下展示 Chain 工作原理.

运行方式:
    cd code/
    LLM_MOCK=1 python tests/_mocks/demo_langchain_basic_chain.py
"""

import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

fake_llm = FakeListChatModel(responses=["GIL 是全局解释器锁, 保证同一时刻只有一个线程执行 Python 字节码。"])
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个 Python 专家。"),
        ("user", "{question}"),
    ]
)
chain = prompt | fake_llm | StrOutputParser()
result = chain.invoke({"question": "什么是 GIL?"})
print(f"Mock answer: {result}")


if __name__ == "__main__":
    print("OK")
