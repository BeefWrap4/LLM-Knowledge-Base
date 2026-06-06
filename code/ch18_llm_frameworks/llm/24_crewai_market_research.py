# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.6.3 CrewAI 角色分工
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: crewai (mocked structure)
# run: python 24_crewai_market_research.py
# expected_runtime: <1s
# expected_output: crew config dump
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.6.3
# Interview hooks:
#   1. CrewAI 的 Process.sequential 与 Process.hierarchical 有什么区别？
#   2. Task 的 context 字段是如何实现"上游任务输出作为下游输入"的？
"""
CrewAI 实战：市场调研团队 - 离线 mock 结构
"""
# 在真实环境：from crewai import Agent, Task, Crew, Process
# from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# ===== 定义工具（mock）=====
search_tool = {"name": "SerperDevTool"}
scrape_tool = {"name": "ScrapeWebsiteTool"}

# ===== 定义 Agent（明确角色分工）=====
market_researcher = {
    "role": "市场研究员",
    "goal": "收集和分析目标市场的最新信息",
    "backstory": "你是一位经验丰富的市场研究员，拥有15年的行业经验。擅长使用搜索工具快速定位关键信息。",
    "tools": [search_tool, scrape_tool],
    "verbose": True,
    "allow_delegation": False,
}

data_analyst = {
    "role": "数据分析师",
    "goal": "将研究的原始数据转化为可操作的商业洞察",
    "backstory": "你是一位精通数据分析的专家，善于从数字中发现趋势和模式。",
    "tools": [],
    "verbose": True,
    "allow_delegation": False,
}

report_writer = {
    "role": "报告撰写人",
    "goal": "将分析结果整合为专业、易读的市场调研报告",
    "backstory": "你是一位专业的商业报告撰写人，曾为多家500强企业撰写报告。",
    "tools": [],
    "verbose": True,
    "allow_delegation": False,
}

# ===== 定义 Task（明确任务分工）=====
research_task = {
    "description": "研究2025-2026年AI Agent框架市场：1. 识别TOP 5框架 2. 分析各框架市场定位 3. 收集用户评价 4. 整理典型应用案例",
    "expected_output": "一份包含框架概述、数据支持、引用来源的详细研究报告，至少1000字。",
    "agent": market_researcher,
}

analysis_task = {
    "description": "基于市场研究员提供的数据进行分析：对比各框架、识别市场趋势、为不同场景提供框架选型建议、评估学习成本和ROI。",
    "expected_output": "一份包含对比表格、趋势分析和选型建议的分析报告。",
    "agent": data_analyst,
    "context": [research_task],
}

writing_task = {
    "description": "整合研究成果和分析结果，撰写的最终调研报告。",
    "expected_output": "一份结构完整、专业美观的Markdown格式市场调研报告。",
    "agent": report_writer,
    "context": [research_task, analysis_task],
    "output_file": "market_research_report.md",
}

# ===== 创建 Crew 并执行（mock）=====
crew = {
    "agents": [market_researcher, data_analyst, report_writer],
    "tasks": [research_task, analysis_task, writing_task],
    "process": "sequential",  # 顺序执行
    "verbose": True,
}

print("=== CrewAI Crew 配置 ===")
print(f"Agent 数量: {len(crew['agents'])}")
for a in crew["agents"]:
    print(f"  - role={a['role']}, tools={len(a['tools'])}")
print(f"Task 数量: {len(crew['tasks'])}")
for i, t in enumerate(crew["tasks"]):
    print(f"  {i+1}. agent={t['agent']['role']}, deps={len(t.get('context', []))}")
print(f"Process: {crew['process']}")
print("\n实际执行需要真实 LLM API（OPENAI_API_KEY）")

if __name__ == "__main__":
    print("OK")
