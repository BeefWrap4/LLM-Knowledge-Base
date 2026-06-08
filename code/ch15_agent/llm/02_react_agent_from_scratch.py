# ---
import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# chapter: 15
# topic: Agent智能体开发
# section: 15.2.3 手写 ReAct Agent
# difficulty: ⭐⭐⭐⭐⭐
# (file top comment - path setup added below)
# tier: llm
# deps: []
# run: python 02_react_agent_from_scratch.py
# expected_runtime: ~1s (模拟 LLM) / API 调用耗时（真实 OpenAI）
# expected_output: 任务执行步骤和最终答案
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.2.3-手写-React-Agent完整实战
# Interview hooks:
#   1. ReAct 循环中 Thought / Action / Observation 各自承担什么职责？
#   2. 如何从 LLM 文本输出中可靠地解析 Action？(正则 + 边界检查)
#   3. ReAct 与 Function Calling 的本质区别？ReAct 是思想、Function Calling 是工程化实现。
"""
从零实现 ReAct Agent - 完整可运行代码
"""
import os
import re
from collections.abc import Callable


class Tool:
    """工具基类"""

    def __init__(self, name: str, description: str, func: Callable, params_schema: dict):
        self.name = name
        self.description = description
        self.func = func
        self.params_schema = params_schema

    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"错误：{str(e)}"

    def to_prompt_format(self) -> str:
        """转换为 Prompt 中的工具描述"""
        params_desc = "\n".join(
            [
                f"  - {k}: {v.get('description', v.get('type', 'string'))}"
                for k, v in self.params_schema.get("properties", {}).items()
            ]
        )
        return f"- {self.name}: {self.description}\n参数：\n{params_desc}"


class ReActAgent:
    """
    ReAct Agent 完整实现

    核心循环：Thought → Action → Observation → ... → Final Answer
    """

    def __init__(self, llm_api_key: str = None):
        self.tools: dict[str, Tool] = {}
        self.memory: list[dict] = []  # 历史记录
        self.max_iterations = 10  # 最大迭代次数，防止无限循环
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")

        # ReAct Prompt 模板
        self.react_prompt_template = """你是一个智能助手，可以通过调用工具来完成任务。

可用工具：
{tools_description}

你必须按照以下格式思考和工作：
Thought: 你的思考过程，分析当前状况和下一步行动
Action: 工具名称(参数1="值1", 参数2="值2")
Observation: 工具返回的结果（由系统自动填入）
... （可以重复多轮 Thought/Action/Observation）
Thought: 任务已完成
Final Answer: 最终答案

---

开始任务！

{history}
Thought: """

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool

    def _build_tools_description(self) -> str:
        """构建工具描述"""
        return "\n".join([t.to_prompt_format() for t in self.tools.values()])

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（Wave 17: 改用 UnifiedClient 支持 deepseek/kimi/siliconflow/MiniMax）"""
        try:
            from shared.llm_client import UnifiedClient

            client = UnifiedClient()
            resp = client.chat(
                messages=[
                    {"role": "system", "content": "你是一个严格遵循 ReAct 格式的智能助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                # 注: stop 参数仅 OpenAI/Anthropic 支持, 其他厂商忽略
            )
            return resp.content
        except Exception:
            # 模拟模式（用于演示和测试）
            return self._simulate_llm(prompt)

    def _simulate_llm(self, prompt: str) -> str:
        """LLM 模拟器（用于无 API 时的测试）"""
        history = prompt.split("开始任务！")[-1] if "开始任务！" in prompt else ""

        # 简单规则匹配模拟决策
        if "weather" in history.lower() or "天气" in history or "温度" in history:
            if "Observation" not in history:
                return '我需要先查询天气信息。\nAction: weather_api(city="北京")'
            elif "calculator" not in history and "average" in history.lower() or "平均" in history:
                return '现在计算体温平均值。\nAction: calculator(expression="(36.5+37.0+36.8)/3")'
            else:
                return "所有信息已获取。\nFinal Answer: 北京今天气温为 25°C，天气晴朗。三人体温平均值为 36.77°C。"

        if "search" in history.lower() or "查" in history:
            if "Observation" not in history:
                return '我需要搜索相关信息。\nAction: search(query="Python GIL")'
            else:
                return "已找到相关信息。\nFinal Answer: Python GIL（全局解释器锁）是 CPython 中防止多线程并发执行字节码的机制。"

        return '我需要分析当前情况。\nAction: search(query="一般信息")'

    def _parse_action(self, text: str) -> tuple[str, dict] | None:
        """从 LLM 输出解析 Action"""
        # 匹配 Action: tool_name(param1="value", param2="value")
        action_pattern = r"Action:\s*(\w+)\((.*)\)"
        match = re.search(action_pattern, text)

        if not match:
            return None

        tool_name = match.group(1)
        params_str = match.group(2)

        # 解析参数
        params = {}
        # 匹配 key="value" 或 key='value' 或 key=value
        param_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]*))'
        for pmatch in re.finditer(param_pattern, params_str):
            key = pmatch.group(1)
            value = pmatch.group(2) or pmatch.group(3) or pmatch.group(4)
            params[key] = value

        return tool_name, params

    def _extract_final_answer(self, text: str) -> str | None:
        """提取 Final Answer"""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_thought(self, text: str) -> str:
        """提取 Thought"""
        match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def run(self, task: str) -> dict:
        """
        执行 ReAct 循环

        Returns:
            {
                "task": str,
                "final_answer": str,
                "steps": list[dict],
                "iterations": int,
            }
        """
        history = f"任务：{task}\n"
        steps = []

        for i in range(self.max_iterations):
            # 构建完整 Prompt
            prompt = self.react_prompt_template.format(
                tools_description=self._build_tools_description(), history=history
            )

            # 调用 LLM 生成 Thought + Action
            llm_output = self._call_llm(prompt)

            thought = self._extract_thought(llm_output)
            final_answer = self._extract_final_answer(llm_output)

            # 检查是否已有最终答案
            if final_answer:
                steps.append({"type": "final", "thought": thought, "answer": final_answer})
                return {
                    "task": task,
                    "final_answer": final_answer,
                    "steps": steps,
                    "iterations": i + 1,
                }

            # 解析 Action
            action_parsed = self._parse_action(llm_output)

            if not action_parsed:
                steps.append({"type": "error", "output": llm_output, "reason": "无法解析 Action"})
                break

            tool_name, params = action_parsed

            # 执行工具
            if tool_name not in self.tools:
                observation = f"错误：工具 '{tool_name}' 不存在。可用工具：{list(self.tools.keys())}"
            else:
                tool = self.tools[tool_name]
                observation = tool.execute(**params)

            # 记录步骤
            steps.append(
                {
                    "type": "action",
                    "thought": thought,
                    "action": f"{tool_name}({params})",
                    "observation": observation,
                }
            )

            # 更新历史
            history += f"{llm_output}\nObservation: {observation}\n"

        # 超过最大迭代次数
        return {
            "task": task,
            "final_answer": "未能完成任务（达到最大迭代次数）",
            "steps": steps,
            "iterations": self.max_iterations,
        }


# ============ 工具函数定义 ============


def weather_api(city: str, date: str = "今天") -> str:
    """模拟天气查询"""
    weather_db = {
        "北京": {"temp": 25, "condition": "晴", "humidity": "45%"},
        "上海": {"temp": 28, "condition": "多云", "humidity": "65%"},
        "深圳": {"temp": 30, "condition": "小雨", "humidity": "80%"},
    }
    info = weather_db.get(city, {"temp": 22, "condition": "未知", "humidity": "50%"})
    return f"{city}{date}天气：{info['condition']}，气温{info['temp']}°C，湿度{info['humidity']}"


def calculator(expression: str) -> str:
    """安全计算器"""
    # 只允许数字和基本运算符
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def search(query: str) -> str:
    """模拟搜索引擎"""
    knowledge_base = {
        "Python GIL": "Python GIL（全局解释器锁）是 CPython 解释器的机制，确保同一时刻只有一个线程执行 Python 字节码。",
        "RAG": "RAG（检索增强生成）将外部知识检索与大语言模型结合，有效减少模型幻觉。",
        "LoRA": "LoRA（低秩适配）是一种参数高效微调方法，通过低秩矩阵微调大模型。",
    }
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return f"搜索结果：{value}"
    return f"搜索结果：找到关于 '{query}' 的 10 条相关网页..."


# ============ 使用示例 ============


def main():
    """主函数 - 运行 ReAct Agent"""
    agent = ReActAgent()

    # 注册工具
    agent.register_tool(
        Tool(
            name="weather_api",
            description="查询指定城市的天气信息",
            func=weather_api,
            params_schema={
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "日期，如'今天'、'明天'"},
                }
            },
        )
    )
    agent.register_tool(
        Tool(
            name="calculator",
            description="执行数学计算",
            func=calculator,
            params_schema={
                "properties": {"expression": {"type": "string", "description": "数学表达式，如(36.5+37.0)/2"}}
            },
        )
    )
    agent.register_tool(
        Tool(
            name="search",
            description="搜索引擎，查询一般知识",
            func=search,
            params_schema={"properties": {"query": {"type": "string", "description": "搜索关键词"}}},
        )
    )

    # 执行任务
    task = "查询北京今天天气，然后计算36.5、37.0、36.8的平均值"
    result = agent.run(task)

    print(f"任务：{result['task']}")
    print(f"最终答案：{result['final_answer']}")
    print(f"迭代次数：{result['iterations']}")
    print("\n详细步骤：")
    for i, step in enumerate(result["steps"], 1):
        print(f"\n--- 步骤 {i} ---")
        if step["type"] == "action":
            print(f"Thought: {step['thought']}")
            print(f"Action: {step['action']}")
            print(f"Observation: {step['observation']}")
        elif step["type"] == "final":
            print(f"Thought: {step['thought']}")
            print(f"Final Answer: {step['answer']}")
    print("\nOK")


if __name__ == "__main__":
    main()
