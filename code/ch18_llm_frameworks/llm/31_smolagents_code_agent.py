# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.8.6 Smolagents
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: smolagents (mocked structure)
# run: python 31_smolagents_code_agent.py
# expected_runtime: <1s
# expected_output: code agent demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.8.6
# Interview hooks:
#   1. Smolagents 的 "Code Agent" 与传统 JSON tool calls 有什么区别？各自优劣？
#   2. 沙箱执行（E2BSandbox）如何保证不可信代码的安全运行？
"""
Smolagents 实战：极简 code-agent - 离线 mock 结构
"""
# 真实环境: from smolagents import CodeAgent, HfApiModel, tool

# ===== 工具定义 =====
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}: 晴 25°C"

# HuggingFace Inference API 上的 Qwen2.5
model_config = {
    "model_id": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "HfApiModel",
}

agent_config = {
    "tools": [get_weather],
    "model": model_config,
    "max_steps": 5,
}

# Agent 内部会写代码：result = get_weather("北京")
def mock_run(query: str) -> str:
    """模拟 CodeAgent.run 的输出"""
    # 真实 CodeAgent 会让 LLM 写出 Python 代码并执行
    generated_code = """
# Agent 生成的代码（mock）
beijing_weather = get_weather("北京")
shanghai_weather = get_weather("上海")
if "晴" in beijing_weather and "晴" in shanghai_weather:
    result = "北京和上海都适合户外运动"
else:
    result = "需要查看更多天气信息"
"""
    return f"执行结果: {generated_code.strip().split(chr(10))[-1]}\n\n(由 CodeAgent 自动生成)"

print("=== Smolagents Code Agent ===")
print(f"工具: {[t.__name__ for t in agent_config['tools']]}")
print(f"模型: {agent_config['model']['model_id']}")
print(f"最大步骤: {agent_config['max_steps']}")

result = mock_run("查询北京和上海的天气，并告诉我哪个更适合户外运动。")
print(f"\n=== 输出 ===\n{result}")

if __name__ == "__main__":
    print("OK")
