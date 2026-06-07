# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.2.2 Few-shot-CoT
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 05_few_shot_cot.py
# expected_runtime: <1s
# expected_output: 打印 Few-shot CoT Prompt 模板及替换后的具体 prompt
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.2.2
# Interview hooks:
# - Few-shot-CoT 中的示例顺序为何"难度递增"更佳？
# - CoT 推理步骤越多越好吗？(过长可能引入累积错误)
# - 如何用 self-consistency 进一步提升 Few-shot-CoT 准确率？

# Few-shot-CoT 示例：数学推理
few_shot_cot_prompt = """
请按照示例中的推理方式，逐步解答问题。

Q: 小明有 24 颗糖，给了小红 8 颗，然后又买了 15 颗。现在有多少颗？
A: 小明开始有 24 颗糖。给了小红 8 颗后，剩下 24 - 8 = 16 颗。
   然后又买了 15 颗，所以现在有 16 + 15 = 31 颗。
   答案是 31。

Q: 一本书 120 页，第一天看了 1/3，第二天看了剩下的 1/4。还剩多少页？
A: 第一天看了 120 × 1/3 = 40 页，剩下 120 - 40 = 80 页。
   第二天看了 80 × 1/4 = 20 页。
   还剩 80 - 20 = 60 页。
   答案是 60。

Q: {question}
A: """

question = "一个水池有甲、乙两个进水管。甲管单独注满需 6 小时，乙管单独注满需 4 小时。两管同时开，几小时注满？"


if __name__ == "__main__":
    print("===== Few-shot CoT 模板 =====")
    print(few_shot_cot_prompt)
    print("===== 替换后的实际 Prompt =====")
    print(few_shot_cot_prompt.format(question=question))
    print("\n[期望推理] 1/(1/6 + 1/4) = 1 / (5/12) = 12/5 = 2.4 小时")
