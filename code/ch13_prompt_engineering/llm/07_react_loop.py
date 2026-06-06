# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.2.4 ReAct
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖 (LLM 调用使用 mock)
# run: python 07_react_loop.py
# expected_runtime: <1s
# expected_output: 打印 ReAct 循环执行轨迹与最终答案
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.2.4
# Interview hooks:
# - ReAct 中 Thought / Action / Observation 三者各自的作用？
# - 为何需要 max_steps 上限？(避免无限循环)
# - ReAct 与函数调用 (function calling) 的关系？

import re

# ReAct Prompt 模板
REACT_PROMPT_TEMPLATE = """回答以下问题，你可以使用以下工具：

工具：
- search(query): 搜索引擎，返回网页摘要
- calculator(expression): 计算器，执行数学运算
- wikipedia(topic): 维基百科查询

请按照以下格式回答：
Thought: 你的思考过程
Action: 工具名称(参数)
Observation: 工具返回的结果
...（以上 Thought/Action/Observation 可重复多轮）...
Thought: 最终结论
Final Answer: 最终答案

---

问题：{question}
"""


def call_llm(history: str) -> str:
    """模拟 LLM：根据 history 中已有 Observation 推进下一步。"""
    if "Observation:" not in history:
        return ("Thought: 我需要先用计算器算一下\n"
                "Action: calculator(2*21)\n")
    if "Final Answer" not in history:
        return ("Thought: 已得到计算结果，可以输出答案\n"
                "Final Answer: 42")
    return ""


def execute_react(question: str, tools: dict, max_steps: int = 5) -> str:
    """
    ReAct 执行循环

    Args:
        question: 用户问题
        tools: {"tool_name": callable, ...}
        max_steps: 最大步数，防止无限循环
    """
    history = REACT_PROMPT_TEMPLATE.format(question=question)

    for step in range(max_steps):
        # 调用 LLM 生成下一步
        response = call_llm(history)

        # 解析 Thought 和 Action
        thought_match = re.search(r"Thought:\s*(.+)", response)
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", response)
        final_match = re.search(r"Final Answer:\s*(.+)", response)

        if final_match:
            history += "\n" + response
            print(f"[step {step}] {response.strip()}")
            return final_match.group(1)

        if action_match:
            tool_name = action_match.group(1)
            tool_arg = action_match.group(2)

            # 执行工具
            if tool_name in tools:
                observation = tools[tool_name](tool_arg)
                history += f"\n{response}\nObservation: {observation}\n"
                print(f"[step {step}] {response.strip()}\nObservation: {observation}")
            else:
                history += f"\n{response}\nObservation: 错误：工具 {tool_name} 不存在\n"

    return "超出最大步数限制，未能完成回答。"


if __name__ == "__main__":
    # 工具定义示例
    tools = {
        "search": lambda q: f"搜索结果：关于 '{q}' 的信息...",
        "calculator": lambda expr: str(eval(expr)),
    }

    answer = execute_react("2 乘以 21 等于多少？", tools=tools, max_steps=3)
    print(f"\n[Final Answer] {answer}")
    print("OK")
