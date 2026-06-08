# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.7.1 智能客服 Agent 完整实现
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: [openai]  # 真实运行需要，mock 模式不需要
# run: export OPENAI_API_KEY=sk-xxx && python 09_customer_service_agent.py
# expected_runtime: API 调用 2-5s / mock 模式 <1s
# expected_output: 三种场景下客服回复、情感分析、转人工判定
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.7.1-智能客服-Agent-完整实现
# Interview hooks:
#   1. 智能客服的"情感分析 + 转人工"机制怎么设计才不会误判？
#   2. 多轮工具调用后对话历史如何管理？本例的策略是什么？
#   3. 客服系统里"敏感操作"（大额退款）二次确认的工程实现？

"""
智能客服 Agent - 完整实战
集成：ReAct + Function Calling + RAG + 记忆管理
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

try:
    import openai

    HAS_OPENAI = True
except Exception:  # pragma: no cover
    HAS_OPENAI = False


@dataclass
class CustomerServiceAgent:
    """
    智能客服 Agent

    能力：
    - 公司政策问答（RAG 知识库）
    - 订单查询（数据库工具）
    - 情感分析与安抚
    - 工单创建与转人工
    """

    api_key: str
    model: str = "gpt-4"
    conversation: list = field(default_factory=list)
    escalation_threshold: float = 0.8  # 转人工阈值
    mock: bool = True  # 默认 mock 模式

    def __post_init__(self):
        # 是否进入 mock：缺 key 或显式开启 mock
        self.mock = self.mock or not HAS_OPENAI or not (self.api_key or os.getenv("OPENAI_API_KEY"))
        if not self.mock and HAS_OPENAI:
            self.client = openai.OpenAI(api_key=self.api_key)
        self.tools = self._define_tools()
        self.system_prompt = self._build_system_prompt()

    def _define_tools(self) -> list:
        """定义客服 Agent 可用的工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_policy",
                    "description": "查询公司政策（如退换货、运费、会员权益等）",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string", "description": "政策主题"}},
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_order",
                    "description": "查询订单信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "user_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "创建工单，转交人工客服",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "issue": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "urgent"],
                            },
                        },
                        "required": ["user_id", "issue", "priority"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "refund_request",
                    "description": "处理退款申请",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            },
        ]

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是某电商平台的智能客服助手「小智」。

核心原则：
1. 专业：准确解答用户问题，不清楚时主动查询政策
2. 共情：用户情绪激动时先安抚，再解决问题
3. 边界：涉及敏感操作（大额退款、投诉）主动转人工
4. 效率：优先用工具查询，不要编造信息

工作流程：
1. 理解用户意图
2. 如需查询信息，调用对应工具
3. 基于查询结果给出准确回答
4. 判断是否需要转人工（用户情绪极差/问题超出能力范围）

当前时间：{time}""".format(time=datetime.now().strftime("%Y-%m-%d %H:%M"))

    def analyze_sentiment(self, message: str) -> dict:
        """情感分析（mock + 真实双模式）"""
        if self.mock:
            # 极简规则匹配：含强负面词则高强度
            negative = ["垃圾", "投诉", "差评", "退款", "欺骗"]
            angry = ["滚", "废物", "去死"]
            intensity = (
                0.9
                if any(w in message for w in angry)
                else 0.6
                if any(w in message for w in negative)
                else 0.2
            )
            sentiment = "angry" if intensity > 0.8 else "neutral"
            return {
                "sentiment": sentiment,
                "intensity": intensity,
                "key_concerns": [message[:20]],
                "needs_escalation": intensity > self.escalation_threshold,
            }
        prompt = f"""分析以下用户消息的情感倾向，输出 JSON：
{{
    "sentiment": "positive/neutral/negative/angry",
    "intensity": 0-1,  // 情绪强度
    "key_concerns": ["用户关注的主要问题"],
    "needs_escalation": true/false  // 是否需要立即转人工
}}

消息：{message}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception:
            return {"sentiment": "neutral", "intensity": 0.5, "needs_escalation": False}

    def handle(self, user_message: str, user_id: str = "anonymous") -> dict:
        """
        处理用户消息
        """
        # 情感分析
        sentiment = self.analyze_sentiment(user_message)

        # 情绪激烈，直接转人工
        if sentiment.get("needs_escalation") or sentiment.get("intensity", 0) > self.escalation_threshold:
            return {
                "response": "非常抱歉给您带来不好的体验，我立即为您转接人工客服 specialist。",
                "actions": [{"type": "escalate", "reason": "high_emotion"}],
                "escalated": True,
                "sentiment": sentiment,
            }

        # mock 模式下的简化决策
        if self.mock:
            tool_name = (
                "query_policy"
                if any(k in user_message for k in ["退换", "运费", "会员"])
                else "query_order"
                if "订单" in user_message
                else None
            )
            response_text = (
                self._execute_tool(tool_name, {"topic": user_message, "order_id": user_message}, user_id)
                if tool_name
                else "您好，请告诉我您想咨询的具体问题（如退换货政策、订单状态等）。"
            )
            self.conversation.append({"role": "user", "content": user_message})
            self.conversation.append({"role": "assistant", "content": response_text})
            return {
                "response": response_text,
                "actions": [{"tool": tool_name}] if tool_name else [],
                "escalated": False,
                "sentiment": sentiment,
            }

        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation,
            {"role": "user", "content": user_message},
        ]

        actions = []
        max_rounds = 3

        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 无需工具调用
            if not message.tool_calls:
                self.conversation.append({"role": "user", "content": user_message})
                self.conversation.append({"role": "assistant", "content": message.content})
                if len(self.conversation) > 20:
                    self.conversation = self.conversation[-20:]

                return {
                    "response": message.content,
                    "actions": actions,
                    "escalated": False,
                    "sentiment": sentiment,
                }

            # 处理工具调用
            tool_results = []
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                result = self._execute_tool(func_name, func_args, user_id)
                tool_results.append(
                    {
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "content": str(result),
                    }
                )
                actions.append({"tool": func_name, "args": func_args, "result": result})

            messages.append(
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
            messages.extend(tool_results)

        return {
            "response": "抱歉，处理时间较长，我为您转接人工客服。",
            "actions": actions + [{"type": "escalate", "reason": "max_rounds"}],
            "escalated": True,
            "sentiment": sentiment,
        }

    def _execute_tool(self, name: str | None, args: dict, user_id: str) -> str:
        """执行工具调用（模拟实现）"""
        if name == "query_policy":
            policies = {
                "退换货": "7天无理由退货，15天换货。商品需保持原状。",
                "运费": "满99包邮，不满收取6元运费。偏远地区除外。",
                "会员": "VIP会员享95折，每月送运费券3张。",
            }
            topic = args.get("topic", "")
            for key, value in policies.items():
                if key in topic:
                    return value
            return f"关于'{topic}'的政策：请参考官网帮助中心。"

        elif name == "query_order":
            order_id = args.get("order_id", "")
            return f"订单 {order_id}：已发货，预计 2-3 天到达。"

        elif name == "create_ticket":
            return f"工单 #{hash(str(args)) % 10000} 已创建，专人将在 10 分钟内联系您。"

        elif name == "refund_request":
            order_id = args.get("order_id", "")
            return f"退款申请已提交（订单 {order_id}），审核需要 1-3 个工作日。"

        return "工具执行成功"


# ============ 使用示例 ============


def demo():
    """智能客服 Agent 演示"""
    agent = CustomerServiceAgent(api_key="your-api-key", mock=True)

    # 场景1：普通政策咨询
    result1 = agent.handle("你们退换货政策是什么？")
    print("用户：你们退换货政策是什么？")
    print(f"客服：{result1['response']}")
    print(f"情感：{result1['sentiment']['sentiment']}, 强度：{result1['sentiment']['intensity']}")
    print()

    # 场景2：订单查询
    result2 = agent.handle("帮我查一下订单 #12345")
    print("用户：帮我查一下订单 #12345")
    print(f"客服：{result2['response']}")
    print(f"执行操作：{result2['actions']}")
    print()

    # 场景3：情绪激动的用户
    result3 = agent.handle("你们这是什么垃圾服务！我的货都丢了一周了！我要投诉！")
    print("用户：你们这是什么垃圾服务！我的货都丢了一周了！")
    print(f"客服：{result3['response']}")
    print(f"是否转人工：{result3['escalated']}")
    print("\nOK")


if __name__ == "__main__":
    demo()
