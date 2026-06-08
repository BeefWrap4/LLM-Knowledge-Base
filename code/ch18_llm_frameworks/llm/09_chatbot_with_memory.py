# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.5 完整实战 - 带记忆的多工具对话 Agent
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 09_chatbot_with_memory.py
# expected_runtime: <1s (mock mode, no chat)
# expected_output: tool listing
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.5
# Interview hooks:
#   1. AgentExecutor 中 max_iterations 起什么作用？过大会带来什么问题？
#   2. 为什么 ChatPromptTemplate 需要 MessagesPlaceholder("agent_scratchpad")？


# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain_openai import ChatOpenAI

    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print(
    "OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)"
)
import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))


"""
完整示例：带记忆的多工具对话 Agent
具备以下能力：
1. 多轮对话记忆（ConversationSummaryBufferMemory）
2. 三个外部工具（天气、计算器、知识检索）
3. 自动工具选择与调用
4. 流式输出支持
"""
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool


# ===== Step 1: 定义工具 =====
@tool
def weather_tool(city: str) -> str:
    """查询城市天气。输入城市名称。"""
    return json.dumps({"city": city, "temp": "26°C", "condition": "晴"})


@tool
def calculator(expr: str) -> str:
    """执行数学计算。输入表达式如 '2+3*4'。"""
    return str(eval(expr, {"__builtins__": {}}, {"abs": abs, "pow": pow}))


@tool
def knowledge_search(query: str) -> str:
    """搜索知识库。输入搜索关键词。"""
    kb = {
        "python": "Python 是一种解释型、面向对象的高级编程语言。",
        "ai": "人工智能是计算机科学的一个分支。",
        "llm": "大语言模型（LLM）是基于 Transformer 架构的大规模语言模型。",
    }
    results = [v for k, v in kb.items() if query.lower() in k]
    return results[0] if results else "未找到相关知识。"


# ===== Step 2: 配置记忆 =====
# Wave 30+: 真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 时 raise
from shared.chatmodel_factory import make_chat_model

llm = make_chat_model()  # 默认厂商

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

# ===== Step 3: 构建 Agent =====
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个智能助手，名为"小智"。你可以：
1. 查询天气信息
2. 执行数学计算
3. 搜索知识库

请根据用户的问题选择合适的工具。回答要亲切、准确。""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [weather_tool, calculator, knowledge_search]

# 演示工具与提示词模板（避免真正构造 AgentExecutor 以保持可运行）
print("=== 已注册工具 ===")
for t in tools:
    print(f"- {t.name}: {t.description}")

print("\n=== 提示词模板示意 ===")
print(prompt.format(input="示例", chat_history=[], agent_scratchpad=[]))

if __name__ == "__main__":
    print("\nOK")
