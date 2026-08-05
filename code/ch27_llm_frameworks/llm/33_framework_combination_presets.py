# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.framework_combination_presets
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: (none - pure stdlib)
# run: python 33_framework_combination_presets.py
# expected_runtime: <1s
# expected_output: combination plans
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. 如何评估一个框架组合方案的合理性？有哪些关键指标？
#   2. 全栈自研 vs 低代码优先：哪些场景必须自研？
# 推荐框架组合方案

# 方案 A：最强组合（全栈自研）
SCENARIO_A = {
    "描述": "企业级全栈大模型应用",
    "组合": {
        "微调": "LLaMA-Factory（微调领域模型）",
        "RAG": "LlamaIndex（构建知识库检索）",
        "Agent": "LangGraph（复杂 Agent 工作流）",
        "编排": "LangChain（胶水代码和工具集成）",
        "部署": "vLLM + FastAPI（高性能推理服务）",
    },
}

# 方案 B：快速落地（低代码优先）
SCENARIO_B = {
    "描述": "中小企业快速搭建 AI 应用",
    "组合": {
        "平台": "Dify（可视化搭建 + 知识库 + Agent）",
        "微调": "LLaMA-Factory（按需微调后导入 Dify）",
        "扩展": "Dify 插件系统（自定义工具）",
    },
}

# 方案 C：研究探索（最强灵活）
SCENARIO_C = {
    "描述": "学术研究 + 原型探索",
    "组合": {
        "多Agent": "AutoGen / CrewAI（根据场景选择）",
        "数据": "LlamaIndex（知识检索）",
        "实验": "Jupyter + LangChain（快速迭代）",
    },
}

# 方案 D：企业知识库（知识管理驱动）
SCENARIO_D = {
    "描述": "企业文档智能问答",
    "组合": {
        "核心": "LlamaIndex（文档索引 + 检索 + 问答）",
        "界面": "Dify（内置 LlamaIndex 或 API 对接）",
        "优化": "微调 Embedding 模型（LLaMA-Factory）",
    },
}

import json

plans = {
    "方案A_全栈自研": SCENARIO_A,
    "方案B_快速落地": SCENARIO_B,
    "方案C_研究探索": SCENARIO_C,
    "方案D_企业知识库": SCENARIO_D,
}
print(json.dumps(plans, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    print("OK")
