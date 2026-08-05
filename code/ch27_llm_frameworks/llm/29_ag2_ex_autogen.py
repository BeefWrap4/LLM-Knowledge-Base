# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.ag2_ex_autogen
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: ag2 (mocked structure)
# run: python 29_ag2_ex_autogen.py
# expected_runtime: <1s
# expected_output: AG2 config dump
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. AG2 与原 AutoGen 0.2.x 有什么 API 层面的变化？迁移成本如何？
#   2. AG2 的 round_robin 与 auto 发言选择各适用于什么场景？
"""
AG2 实战：现代化多 Agent 协作 - 离线 mock 结构
"""
# 真实环境: from ag2 import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


# 配置 LLM
llm_config = {
    "model": "<inject-current-model-in-real-integration>",
    "api_key": "<inject-from-secret-store-in-real-integration>",
}

# ===== 定义 Agents =====
planner = {
    "name": "Planner",
    "system_message": "你是项目规划师，拆解任务为子任务。",
    "llm_config": llm_config,
}

coder = {
    "name": "Coder",
    "system_message": "你是 Python 专家，实现子任务代码。",
    "llm_config": llm_config,
}

reviewer = {
    "name": "Reviewer",
    "system_message": "你是代码审查员，确保代码质量。",
    "llm_config": llm_config,
}

user = {
    "name": "User",
    "code_execution_config": {"work_dir": "coding"},
    "human_input_mode": "NEVER",
}

# ===== GroupChat =====
chat = {
    "agents": [user, planner, coder, reviewer],
    "speaker_selection_method": "round_robin",  # 确定性发言顺序
    "max_round": 8,
}
manager = {
    "groupchat": chat,
    "llm_config": llm_config,
}

print("=== AG2 GroupChat 配置 ===")
for a in chat["agents"]:
    print(f"  - {a['name']}: {a.get('system_message', a.get('human_input_mode', ''))[:60]}")
print(f"\n发言选择: {chat['speaker_selection_method']} (确定性顺序)")
print(f"最大轮次: {chat['max_round']}")
print("\n初始任务: 实现一个分布式任务调度系统")

if __name__ == "__main__":
    print("OK")
