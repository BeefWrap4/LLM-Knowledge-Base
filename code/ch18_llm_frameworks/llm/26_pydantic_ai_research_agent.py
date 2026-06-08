# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.8.1 Pydantic AI
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: pydantic-ai
# run: python 26_pydantic_ai_research_agent.py
# expected_runtime: <1s
# expected_output: pydantic model definition
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.8.1
# Interview hooks:
#   1. Pydantic AI 相比 LangChain 的核心优势是什么？为什么 2026 年被认为是首选？
#   2. Pydantic AI 的"结构化输出"如何在工程上保证类型安全？
"""
Pydantic AI 实战：类型安全的研究助手 - 离线 mock 模式
"""

# 真实环境: from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

from pydantic import BaseModel, Field


# ===== 1. 用 Pydantic 模型声明 Agent 输出 =====
class ResearchReport(BaseModel):
    """研究结果的结构化输出"""

    summary: str = Field(description="一句话总结")
    key_points: list[str] = Field(description="3-5 个关键点")
    sources: list[str] = Field(description="引用来源列表")
    confidence: float = Field(ge=0, le=1, description="置信度 0-1")


# ===== 2. 通过依赖注入传递上下文 =====
@dataclass
class Deps:
    user_id: str
    api_key: str


# ===== 3. 定义 Agent（mock 模式：仅展示 API）=====
# 真实代码:
# research_agent = Agent(
#     model="openai:gpt-4o",
#     result_type=ResearchReport,  # 强制结构化输出
#     system_prompt="你是一个严谨的研究助手，输出必须可验证。",
#     deps_type=Deps,
# )
#
# @research_agent.tool
# async def web_search(ctx: RunContext[Deps], query: str) -> str:
#     """联网搜索工具（自动注册为 LLM 可用工具）"""
#     return f"[{query}] 的模拟搜索结果：..."


# ===== 4. mock 运行 =====
def mock_run(query: str, deps: Deps) -> ResearchReport:
    """模拟 Pydantic AI 的运行结果"""
    return ResearchReport(
        summary=f"基于用户 {deps.user_id} 的查询 '{query}' 的研究总结。",
        key_points=[
            "LangChain 已转向编排与集成层",
            "LangGraph 主导复杂 Agent 工作流",
            "Pydantic AI 在类型安全上领先",
        ],
        sources=["官方文档", "GitHub 仓库", "社区博客"],
        confidence=0.85,
    )


# 演示
deps = Deps(user_id="alice", api_key="sk-...")
report = mock_run("请调研 2026 年 LangChain 的市场份额变化", deps)
print("类型安全：IDE 自动补全、运行时校验")
print(f"summary: {report.summary}")
print(f"key_points: {report.key_points}")
print(f"confidence: {report.confidence}")

if __name__ == "__main__":
    print("OK")
