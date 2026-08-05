# ---
# chapter: 24
# topic: Agent 工作流编排与多智能体
# topic_id: llm_frameworks.autogen_code_review
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: autogen (mocked structure)
# run: python 23_autogen_code_review.py
# expected_runtime: <1s
# expected_output: agent config dump
# ---
# See: ../../../24_Agent工作流编排与多智能体.md
# Interview hooks:
#   1. AutoGen 的 GroupChat 与 CrewAI 的 Process 模式各有什么优劣？
#   2. UserProxyAgent 的 human_input_mode 三个选项分别表示什么？
"""
AutoGen 实战：多 Agent 代码审查系统 - 离线 mock 模式
"""
# 在真实环境使用：from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


# ===== 配置 LLM =====
config_list = [
    {
        "model": "<inject-current-model-in-real-integration>",
        "api_key": "<inject-from-secret-store-in-real-integration>",
    }
]

llm_config = {
    "config_list": config_list,
    "timeout": 120,
}

# ===== 定义 Agent 角色（用 dict 模拟 Agent 对象）=====
agents = {
    "user_proxy": {
        "name": "user_proxy",
        "system_message": "你是一个开发者，需要审查代码。",
        "human_input_mode": "TERMINATE",  # 只在终止时请求人工输入
        "max_consecutive_auto_reply": 5,
        "code_execution_config": {"work_dir": "coding", "use_docker": False},
    },
    "coder": {
        "name": "coder",
        "system_message": "你是一个资深Python程序员。编写清晰、高效、有注释的代码。使用类型标注和文档字符串。遵循 PEP 8 规范。",
        "llm_config": llm_config,
    },
    "reviewer": {
        "name": "reviewer",
        "system_message": "你是一个严格的代码审查员。检查代码的：1. 正确性 2. 安全性 3. 性能 4. 可读性。提供具体的改进建议。",
        "llm_config": llm_config,
    },
    "tester": {
        "name": "tester",
        "system_message": "你是一个测试工程师。为代码编写全面的单元测试。覆盖正常情况、边界情况和异常情况。",
        "llm_config": llm_config,
    },
}

# ===== 创建群组对话（mock 结构）=====
groupchat = {
    "agents": list(agents.values()),
    "messages": [],
    "max_round": 12,
    "speaker_selection_method": "auto",  # 自动选择下一个发言者
}

manager = {
    "groupchat": groupchat,
    "llm_config": llm_config,
}

# ===== 启动对话（mock）=====
INITIAL_MESSAGE = """
请实现一个 LRU Cache（最近最少使用缓存），要求：
1. 支持 get(key) 和 put(key, value) 操作
2. 时间复杂度 O(1)
3. 固定容量，满时淘汰最久未使用的项
4. 线程安全
"""

print("=== AutoGen 多 Agent 配置 ===")
for k, v in agents.items():
    print(f"\n[{k}]")
    for kk, vv in v.items():
        print(f"  {kk}: {str(vv)[:80]}")
print(f"\n最大轮次: {groupchat['max_round']}")
print(f"发言选择: {groupchat['speaker_selection_method']}")
print(f"\n初始任务:\n{INITIAL_MESSAGE}")

if __name__ == "__main__":
    print("OK")
