# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.3.2 Self-Instruct 方法
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 04_self_instruct.py
# expected_runtime: <2s
# expected_output: 生成的指令列表
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. Self-Instruct 的核心流程是什么？为什么要用 8 条种子做 few-shot？
#   2. Self-Instruct 的局限性有哪些？为什么生成的指令容易陷入模式重复？
#   3. Self-Instruct 与 Evol-Instruct 在数据合成策略上有什么本质差异？

import json
from typing import List, Dict, Callable

# 种子指令模板
SEED_INSTRUCTIONS = [
    {"instruction": "将以下句子翻译成英文", "input": "今天天气真好", "output": "The weather is really nice today."},
    {"instruction": "用一句话总结以下段落的核心观点", "input": "...", "output": "..."},
    {"instruction": "生成一个随机密码", "input": "", "output": "K9#mP2xL@qR5"},
]

INSTRUCTION_GEN_PROMPT = """你是一个数据标注专家。请根据以下已有的指令示例，生成5条新的、多样化的任务指令。

要求：
1. 指令应该覆盖不同类型的任务（翻译、摘要、分类、代码生成、推理等）
2. 指令应该具体、清晰、可执行
3. 避免与已有指令语义重复
4. 难度应该有所梯度

已有指令示例：
{examples}

请生成5条新指令（每行一条，以 "指令N：" 开头）："""


def generate_instructions(
    seed_instructions: List[Dict],
    llm_callable: Callable[[str], str],
    num_to_generate: int = 5
) -> List[str]:
    """
    使用 Self-Instruct 方法生成新指令

    Args:
        seed_instructions: 种子指令列表
        llm_callable: LLM 调用函数 (prompt -> response)
        num_to_generate: 需要生成的指令数量
    """
    examples_text = "\n".join([
        f"{i + 1}. {inst['instruction']}"
        for i, inst in enumerate(seed_instructions[:8])
    ])

    prompt = INSTRUCTION_GEN_PROMPT.format(examples=examples_text)
    response = llm_callable(prompt)

    # 解析生成的指令
    new_instructions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and any(line.startswith(f"指令{i}") for i in range(1, 10)):
            instruction = line.split('：', 1)[-1].strip().strip('"').strip("'")
            if instruction:
                new_instructions.append(instruction)

    return new_instructions[:num_to_generate]


# 模拟 LLM 调用（实际使用时替换为真实API调用）
def mock_llm_call(prompt: str) -> str:
    """Mock LLM 调用 - 实际使用可接入 OpenAI / Anthropic / vLLM"""
    return """指令1：分析以下代码的时间复杂度
指令2：将给定的JSON数据转换为CSV格式
指令3：写一首关于人工智能的七言绝句
指令4：判断以下评论的情感倾向（正面/负面/中性）
指令5：根据用户需求生成一个SQL查询语句"""


def main():
    new_instructions = generate_instructions(SEED_INSTRUCTIONS, mock_llm_call)
    print("生成的指令:", json.dumps(new_instructions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    print("OK")
