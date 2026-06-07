# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.4 Tool 定义与使用
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 08_tool_definition_agent.py
# expected_runtime: <1s (mock mode)
# expected_output: tool use result
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.4
# Interview hooks:
#   1. @tool 装饰器是如何将函数注册为 LLM 可用工具的？
#   2. create_openai_functions_agent 与 create_react_agent 的区别是什么？


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print("OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)")
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json

# ===== 方式1: 使用 @tool 装饰器 =====
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。参数 city 为城市名称（中文）。"""
    # 模拟 API 调用
    weather_data = {
        "北京": "晴，25°C，湿度45%",
        "上海": "多云，28°C，湿度60%",
        "深圳": "阵雨，30°C，湿度80%",
    }
    return weather_data.get(city, f"未找到{city}的天气数据")

@tool
def calculate(expression: str) -> str:
    """执行数学计算。参数 expression 为数学表达式字符串，如 '2+3*4'。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def search_database(query: str, limit: int = 5) -> str:
    """在数据库中搜索信息。参数 query 为搜索关键词，limit 为返回结果数量。"""
    # 模拟数据库查询
    mock_db = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架...",
        "python": "Python 3.14 预计于 2025 年发布...",
        "gpt": "GPT-4o 是 OpenAI 的多模态模型...",
    }
    results = []
    for k, v in mock_db.items():
        if query.lower() in k or query.lower() in v:
            results.append(f"[{k}]: {v[:100]}...")
    return "\n".join(results[:limit]) or "未找到匹配结果"

# 工具列表
tools = [get_weather, calculate, search_database]

# 直接演示工具调用（不依赖 LLM 决策）
print("=== 工具自测 ===")
print(get_weather.invoke({"city": "北京"}))
print(calculate.invoke({"expression": "123 * 456"}))
print(search_database.invoke({"query": "LangChain"}))

print("\n=== 工具 Schema ===")
for t in tools:
    print(f"- {t.name}: {t.description}")

if __name__ == "__main__":
    print("OK")