# ---
# chapter: 29
# topic: 大模型数据工程
# topic_id: data_engineering.evol_instruct
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 05_evol_instruct.py
# expected_runtime: <2s
# expected_output: 演化后的 prompt 字符串
# ---
# See: ../../../29_大模型数据工程.md
#
# Interview hooks:
#   1. Evol-Instruct 的"深度演化 (In-Depth)"和"广度演化 (In-Breadth)"区别是什么？
#   2. 演化操作（添加约束/深化推理/多步推理）如何影响 SFT 数据的难度分布？
#   3. WizardLM 是如何基于 Evol-Instruct 超过 Alpaca 的？数据演化的边界在哪里？

from collections.abc import Callable

EVOLVE_PROMPTS = {
    "in_depth": """你是指令复杂度演化的专家。请将以下简单指令演化为更深入、要求更高推理能力的版本。

演化策略（请至少使用3种）：
- 增加更多约束条件（如时间/空间限制、特定格式要求）
- 要求逐步推理（Chain-of-Thought）
- 增加对边界情况/异常情况的处理
- 引入多步推理链条

原始指令: {instruction}
演化后的指令:""",
    "in_breadth": """你是指令多样性专家。请基于以下指令，创建一个涵盖更广知识面或技能范围的新指令。

演化策略（请至少使用2种）：
- 扩展到相关但不相同的领域
- 增加子任务或多角度要求
- 引入跨领域知识需求

原始指令: {instruction}
演化后的指令:""",
    "add_constraints": """请为以下指令添加2-3个合理但具有挑战性的约束条件。

原始指令: {instruction}
增加约束后的指令:""",
}


def evolve_instruction(
    instruction: str,
    evolve_type: str = "in_depth",
    llm_callable: Callable[[str], str] | None = None,
) -> str:
    """
    对指令进行演化

    Args:
        instruction: 原始指令
        evolve_type: 演化类型 (in_depth / in_breadth / add_constraints)
        llm_callable: LLM 调用函数
    """
    template = EVOLVE_PROMPTS.get(evolve_type, EVOLVE_PROMPTS["in_depth"])
    prompt = template.format(instruction=instruction)
    if llm_callable:
        return llm_callable(prompt).strip()
    return prompt  # 返回 prompt 供外部调用


def main():
    # 演示三种演化类型
    original = "写一个排序函数"
    for etype in ["in_depth", "in_breadth", "add_constraints"]:
        result = evolve_instruction(original, evolve_type=etype, llm_callable=None)
        print(f"--- 演化类型: {etype} ---")
        print(result)
        print()
    print("OK")


if __name__ == "__main__":
    main()
