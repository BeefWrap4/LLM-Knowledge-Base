# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.8.3 OpenAI Agents SDK
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai-agents (mocked structure)
# run: python 28_openai_agents_sdk_handoff.py
# expected_runtime: <1s
# expected_output: agent handoff config
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.8.3
# Interview hooks:
#   1. OpenAI Agents SDK 的 Multi-Agent Handoffs 与 AutoGen GroupChat 的区别？
#   2. SandboxAgent v0.14.0 的沙箱执行对生产环境有什么价值？
"""
OpenAI Agents SDK 实战：多 agent 客服系统 - 离线 mock 结构
"""
# 真实环境: from agents import Agent, Runner, function_tool, handoff
# from agents.extensions.sandbox import SandboxAgent


# ===== 1. 定义工具 =====
def check_order(order_id: str) -> str:
    """查询订单状态"""
    return f"订单 {order_id} 状态：已发货，预计明天到达"


# ===== 2. 定义专业 agent =====
billing_agent = {
    "name": "Billing",
    "instructions": "处理账单、退款、发票问题。如不确定请转交。",
    "model": "gpt-4o",
}

tech_agent = {
    "name": "TechSupport",
    "instructions": "处理技术问题：登录错误、功能 bug。",
    "model": "gpt-4o",
    "tools": [check_order],
}

# ===== 3. 路由 agent：识别意图并 handoff =====
triage_agent = {
    "name": "Triage",
    "instructions": "根据用户问题分诊到 Billing 或 TechSupport。",
    "model": "gpt-4o",
    "handoffs": [
        {"to": "Billing", "tool_description_override": "转账单客服"},
        {"to": "TechSupport", "tool_description_override": "转技术支持"},
    ],
}


# ===== 4. mock 运行 =====
class _MockRunner:
    @staticmethod
    async def run(agent, input, session_id):
        return {
            "final_output": f"（mock）{agent['name']} 处理了: {input}",
            "trace": type("T", (), {"url": f"https://platform.openai.com/traces/{session_id}"})(),
        }


print("=== OpenAI Agents SDK 配置 ===")
print(f"分诊 Agent: {triage_agent['name']} → handoffs: {[h['to'] for h in triage_agent['handoffs']]}")
print(f"  - {billing_agent['name']}: {billing_agent['instructions']}")
print(
    f"  - {tech_agent['name']}: {tech_agent['instructions']} (tools: {[t.__name__ for t in tech_agent['tools']]})"
)

# 同步演示异步运行结果
import asyncio


async def demo():
    result = await _MockRunner.run(
        triage_agent,
        input="我的订单 #12345 一直没收到，我想退款。",
        session_id="user-001",
    )
    print(f"\n最终回复: {result['final_output']}")
    print(f"执行轨迹: {result['trace'].url}")


asyncio.run(demo())

if __name__ == "__main__":
    print("OK")
