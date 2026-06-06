# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.6.4 选型决策
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (none - pure stdlib)
# run: python 25_multi_agent_framework_choice.py
# expected_runtime: <1s
# expected_output: decision routing
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.6.4
# Interview hooks:
#   1. 在多 Agent 框架选型时，需要考虑哪些因素？
#   2. AutoGen 和 CrewAI 的设计哲学有什么不同？
# 决策伪代码
def choose_multi_agent_framework(scenario: str) -> str:
    if "代码生成" in scenario or "数学推理" in scenario:
        return "AutoGen（内置代码执行，推理能力强）"
    elif "商业分析" in scenario or "内容创作" in scenario:
        return "CrewAI（角色分工清晰，上手快）"
    elif "研究探索" in scenario and "需要自由讨论" in scenario:
        return "AutoGen（GroupChat 适合开放讨论）"
    elif "结构化流水线" in scenario and "明确上下游" in scenario:
        return "CrewAI（Process 模式适合流水线）"
    elif "需要人机协同" in scenario:
        return "AutoGen（UserProxyAgent 更成熟）"
    else:
        return "两者均可，建议先试用 CrewAI 上手更快"


# 演示
scenarios = [
    "代码生成：实现 LRU 缓存",
    "商业分析：竞品调研报告",
    "研究探索：需要自由讨论的开放式问题",
    "结构化流水线：明确上下游任务的处理流程",
    "需要人机协同：关键步骤需要人工审批",
    "未知场景",
]
print("=== 框架选型决策 ===")
for s in scenarios:
    print(f"- {s} → {choose_multi_agent_framework(s)}")

if __name__ == "__main__":
    print("OK")
