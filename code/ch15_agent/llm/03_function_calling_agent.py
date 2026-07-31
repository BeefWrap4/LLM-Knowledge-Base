# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.3.3 多工具调用 Agent 实战
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [openai]
# run: export OPENAI_API_KEY=sk-xxx && python 03_function_calling_agent.py
# expected_runtime: API 调用耗时 2-10s（真实 OpenAI）
# expected_output: 多工具组合查询结果；若无 API key 则进入 mock 模式
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.3.3-多工具调用-Agent-实战
# Interview hooks:
#   1. OpenAI Function Calling 的消息流是 tool_call 怎么回传的？(role=tool, tool_call_id)
#   2. 一次请求模型可以并行返回多个 tool_calls 吗？怎么遍历执行？
#   3. tool_choice="auto" 与 "required"、"none" 的区别？
"""
Function Calling 多工具 Agent - 完整实战
"""

import json
import os

try:
    import openai

    HAS_OPENAI = True
except Exception:  # pragma: no cover
    HAS_OPENAI = False


class FunctionCallingAgent:
    """
    基于 OpenAI Function Calling 的 Agent

    核心流程：
    1. 定义 tools（函数 schema）
    2. 发送用户消息 + tools 定义
    3. 如果模型返回 function_call，执行对应函数
    4. 将结果返回给模型，生成最终回答
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        mock: bool = False,
    ):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.6")
        self.model_kwargs = (
            {"reasoning_effort": "none"} if self.model.startswith("gpt-5.6") else {}
        )
        self.tools = []
        self.tool_functions = {}
        self.conversation = []
        self.mock = (
            mock
            or os.environ.get("LLM_MOCK") != "0"
            or not HAS_OPENAI
            or not (api_key or os.getenv("OPENAI_API_KEY"))
        )
        if not self.mock and HAS_OPENAI:
            self.client = openai.OpenAI(api_key=api_key)

    def register_tool(self, name: str, description: str, parameters: dict, func: callable):
        """注册工具"""
        self.tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
        self.tool_functions[name] = func

    def _mock_response(self, user_message: str) -> "openai.types.chat.ChatCompletionMessage":
        """无 API 时的离线模拟"""
        from types import SimpleNamespace

        msg_lower = user_message
        if "天气" in msg_lower:
            tc_args = json.dumps({"city": "北京"})
            tool_call = SimpleNamespace(
                id="mock-1",
                function=SimpleNamespace(name="get_weather", arguments=tc_args),
            )
            content = None
        elif "股价" in msg_lower or "AAPL" in msg_lower:
            tc_args = json.dumps({"symbol": "AAPL"})
            tool_call = SimpleNamespace(
                id="mock-2",
                function=SimpleNamespace(name="get_stock_price", arguments=tc_args),
            )
            content = None
        else:
            tool_call = None
            content = "[Mock] 暂无更多工具需要调用，最终回答：北京天气晴 25°C，AAPL 股价 182.50 USD。"
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                tool_calls=[tool_call] if tool_call else None,
            )
        )
        return SimpleNamespace(choices=[choice])

    def execute(self, user_message: str, max_tool_calls: int = 5) -> str:
        """执行对话循环"""
        self.conversation = [{"role": "user", "content": user_message}]

        for _ in range(max_tool_calls):
            if self.mock:
                response = self._mock_response(user_message)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto",
                    **self.model_kwargs,
                )

            message = response.choices[0].message

            # 检查是否需要调用工具
            if not message.tool_calls:
                # 不需要工具，直接返回回答
                return message.content

            # 记录助手消息（含 tool_calls）
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # 执行所有工具调用
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name in self.tool_functions:
                    result = self.tool_functions[func_name](**func_args)
                else:
                    result = f"错误：工具 {func_name} 不存在"

                # 将工具结果加入对话
                self.conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        # 达到最大工具调用次数，生成最终回答
        if self.mock:
            return "[Mock] 已达到最大工具调用次数，强制返回兜底回答。"
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            **self.model_kwargs,
        )
        return final_response.choices[0].message.content


# ============ 工具函数定义 ============


def get_weather(city: str) -> str:
    """获取天气（模拟）"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
        "深圳": "雷阵雨，29°C",
    }
    return weather_data.get(city, "未知城市")


def get_stock_price(symbol: str) -> str:
    """获取股票价格（模拟）"""
    stocks = {
        "AAPL": "182.50 USD",
        "GOOGL": "142.30 USD",
        "MSFT": "380.20 USD",
        "TSLA": "240.10 USD",
    }
    return stocks.get(symbol.upper(), "未知股票代码")


def search_knowledge(query: str) -> str:
    """知识库搜索（模拟）"""
    kb = {
        "年假": "员工每年享有 15 天带薪年假，入职满 1 年后可申请。",
        "报销": "差旅报销需在出差结束后 30 天内提交，附发票和行程单。",
        "加班": "加班需提前在 OA 系统申请，加班费按法定标准计算。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return f"未找到 '{query}' 的相关政策"


def send_notification(to: str, message: str) -> str:
    """发送通知（模拟）"""
    return f"通知已发送给 {to}: {message}"


# ============ 使用示例 ============


def main():
    agent = FunctionCallingAgent(mock=True)  # 默认 mock 模式避免无 key 报错

    # 注册工具
    agent.register_tool(
        name="get_weather",
        description="获取指定城市的当前天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名称，如北京、上海"}},
            "required": ["city"],
        },
        func=get_weather,
    )

    agent.register_tool(
        name="get_stock_price",
        description="获取指定股票的当前价格",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "股票代码，如 AAPL、GOOGL"}},
            "required": ["symbol"],
        },
        func=get_stock_price,
    )

    agent.register_tool(
        name="search_knowledge",
        description="搜索公司内部知识库",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
        func=search_knowledge,
    )

    agent.register_tool(
        name="send_notification",
        description="发送通知给指定人员",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "接收人"},
                "message": {"type": "string", "description": "通知内容"},
            },
            "required": ["to", "message"],
        },
        func=send_notification,
    )

    # 测试：需要调用多个工具的复杂查询
    query = "北京天气怎么样？顺便帮我查一下 AAPL 的股价。如果天气好，通知小王出门记得带伞。"
    result = agent.execute(query)
    print(f"查询：{query}")
    print(f"结果：{result}")
    print("OK")


if __name__ == "__main__":
    main()
