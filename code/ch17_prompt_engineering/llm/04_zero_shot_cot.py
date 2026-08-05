# ---
# chapter: 17
# topic: Prompt Engineering
# topic_id: prompt_engineering.zero_shot_cot
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 04_zero_shot_cot.py
# expected_runtime: <1s
# expected_output: 打印 Zero-shot-CoT Prompt 与预期推理示例
# ---
# See: ../../../17_Prompt_Engineering.md
# Interview hooks:
# - 为何"Let's think step by step"能显著提升推理准确率？
# - Zero-shot CoT 与 Few-shot CoT 在原理上的差异？
# - CoT 在哪类任务上无效甚至会反向降分？

# Zero-shot-CoT 示例
prompt_zero_shot_cot = """
问题：一个农场有鸡和兔共 35 只，脚共 94 只。鸡和兔各有多少只？

请逐步思考，在最后一行以 "答案：X" 的格式给出结果。
"""

EXPECTED_RESPONSE = """\
设鸡有 x 只，兔有 y 只。
根据题意：x + y = 35
          2x + 4y = 94
从第一式得：x = 35 - y
代入第二式：2(35 - y) + 4y = 94
            70 - 2y + 4y = 94
            2y = 24
            y = 12
所以 x = 35 - 12 = 23
答案：鸡 23 只，兔 12 只
"""


if __name__ == "__main__":
    print("===== Zero-shot-CoT Prompt =====")
    print(prompt_zero_shot_cot)
    print("===== 模型推理示例输出 =====")
    print(EXPECTED_RESPONSE)
    print("OK")
