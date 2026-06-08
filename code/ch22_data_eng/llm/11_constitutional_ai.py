# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.5.7 Constitutional AI 数据生成
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 11_constitutional_ai.py
# expected_runtime: <2s
# expected_output: Constitutional AI 数据（SL-CAI 训练对 + RL-CAI 偏好对）
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. Constitutional AI 两阶段（SL-CAI / RL-CAI）的区别是什么？为什么需要分两阶段？
#   2. Constitution 原则集的设计有哪些隐性陷阱（价值观锁定、原则冲突）？
#   3. CAI 与传统 RLHF 相比，成本降低 10-100 倍的代价是什么？

from collections.abc import Callable

CRITIQUE_PROMPT = """以下是一个 AI 助手对用户问题的回复。请根据宪法原则评判：

{constitution}

用户问题: {user_prompt}
AI 回复: {ai_response}

请分析回复是否违反了任何原则。如有违反，请指出具体问题：
"""

REVISE_PROMPT = """根据以下批评意见，修改 AI 的回复使其符合所有宪法原则：

用户问题: {user_prompt}
原始 AI 回复: {ai_response}
批评意见: {critique}

修改后的回复:
"""


# 默认 Constitution（Anthropic 公开原则节选）
DEFAULT_CONSTITUTION = """
Principle 1: 选择对人类最不有害的回复。
Principle 2: 选择最诚实和最透明的回复。
Principle 3: 选择避免给医疗、法律、财务建议的回复（除非明确请求且有适当免责声明）。
Principle 4: 选择最不带有种族、性别等偏见的回复。
Principle 5: 选择最尊重用户隐私的回复。
"""


def mock_llm_callable(prompt: str) -> str:
    """Mock LLM 调用"""
    if "评判" in prompt or "批评" in prompt or "分析" in prompt:
        return "回复中包含未经核实的医疗建议，违反 Principle 3。建议添加免责声明或建议咨询专业医生。"
    return "感谢您的提问。根据医学常识，这种情况建议您咨询专业医生以获得准确诊断。本回答仅供参考。"


def generate_cai_data(
    user_prompt: str, initial_response: str, constitution: str, llm_callable: Callable[[str], str]
) -> dict:
    """生成 Constitutional AI 训练数据"""
    # 步骤1: 让模型自我批评
    critique_input = CRITIQUE_PROMPT.format(
        constitution=constitution, user_prompt=user_prompt, ai_response=initial_response
    )
    critique = llm_callable(critique_input)

    # 步骤2: 让模型根据批评修正回复
    revise_input = REVISE_PROMPT.format(
        user_prompt=user_prompt, ai_response=initial_response, critique=critique
    )
    revised_response = llm_callable(revise_input)

    return {
        "prompt": user_prompt,
        "original": initial_response,
        "critique": critique,
        "revised": revised_response,
        # SL-CAI 训练数据
        "sft_pair": {"instruction": user_prompt, "output": revised_response},
        # RL-CAI 偏好对（修正版优于原版）
        "preference_pair": {"chosen": revised_response, "rejected": initial_response},
    }


def main():
    user_prompt = "我最近总是头痛，吃什么药比较好？"
    initial_response = "你应该立即服用阿司匹林，每天三次。"

    cai_data = generate_cai_data(user_prompt, initial_response, DEFAULT_CONSTITUTION, mock_llm_callable)
    print("=== Constitutional AI 训练数据 ===")
    print(f"Prompt: {cai_data['prompt']}")
    print(f"\n原始回复: {cai_data['original']}")
    print(f"\n批评意见: {cai_data['critique']}")
    print(f"\n修正后回复: {cai_data['revised']}")
    print(f"\nSL-CAI 训练对 output: {cai_data['sft_pair']['output']}")


if __name__ == "__main__":
    main()
