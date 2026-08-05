# ---
# chapter: 17
# topic: Prompt Engineering
# topic_id: prompt_engineering.structured_prompt_template
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 01_structured_prompt_template.py
# expected_runtime: <1s
# expected_output: 打印结构化分隔的 Prompt 模板字符串
# ---
# See: ../../../17_Prompt_Engineering.md
# Interview hooks:
# - 为什么使用 XML 标签 / Markdown 分隔符可以提升模型理解准确率？
# - 多段约束应该放在 Prompt 开头还是结尾？为什么？
# - 如何让模型严格按指定字数生成文案？

prompt = """
请根据以下产品描述，生成5条营销文案。

<product_description>
产品：智能降噪耳机 Pro
特点：-40dB 主动降噪、40小时续航、Hi-Res 金标认证
目标人群：25-35岁都市白领
</product_description>

<requirements>
- 每条文案不超过 30 字
- 突出续航和降噪两个卖点
- 风格：简洁有力、有记忆点
</requirements>
"""


if __name__ == "__main__":
    print(prompt)
    print(f"\n[Prompt 长度] {len(prompt)} 字符")
    print("OK")
