# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.6.7 Constitutional AI 数据生成
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 11_constitutional_ai.py
# expected_runtime: <2s
# expected_output: Constitutional AI 数据（SL-CAI 训练对 + 待独立评判的 RL-CAI 候选）
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. Constitutional AI 两阶段（SL-CAI / RL-CAI）的区别是什么？为什么需要分两阶段？
#   2. Constitution 原则集的设计有哪些隐性陷阱（价值观锁定、原则冲突）？
#   3. CAI 把部分人类偏好标注转为 AI 反馈后，引入了哪些校准、偏差与审计成本？

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


# 教学用 Constitution（非 Anthropic 当前文档的逐字节选）
DEFAULT_CONSTITUTION = """
Principle 1: 选择对人类最不有害的回复。
Principle 2: 选择最诚实和最透明的回复。
Principle 3: 选择避免给医疗、法律、财务建议的回复（除非明确请求且有适当免责声明）。
Principle 4: 选择最不带有种族、性别等偏见的回复。
Principle 5: 选择最尊重用户隐私的回复。
"""


def mock_llm_callable(prompt: str) -> str:
    """Mock LLM 调用"""
    if "修改后的回复:" in prompt:
        return (
            "头痛原因很多，不能仅凭这段描述安全地推荐具体药物或剂量。"
            "如果症状持续、加重，或伴随突发剧烈头痛、发热、肢体无力等情况，请尽快就医；"
            "其他情况也建议咨询医生或药师。本回答不能替代专业诊断。"
        )
    if "请分析回复是否违反" in prompt:
        return "回复中包含未经核实的医疗建议，违反 Principle 3。建议添加免责声明或建议咨询专业医生。"
    raise ValueError("无法识别的教学 mock prompt")


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
        # 这里只产出待比较候选；原始 RL-CAI 还需按 Constitution 独立评判后才能标注 chosen/rejected。
        "preference_candidates": {
            "response_a": revised_response,
            "response_b": initial_response,
            "label": None,
        },
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
    print("\nRL-CAI 候选尚未标注；需由独立 AI 评判并经过人工校准。")
    assert cai_data["revised"] != cai_data["critique"], "修正答复不得复用批评文本"
    assert cai_data["preference_candidates"]["label"] is None
    print("OK")


if __name__ == "__main__":
    main()
