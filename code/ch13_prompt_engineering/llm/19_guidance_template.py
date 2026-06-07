# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.4 guidance 模板引导生成
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: guidance (可选)
# run: python 19_guidance_template.py
# expected_runtime: <1s (mock)
# expected_output: 打印 guidance 模板字符串与说明
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.4
# Interview hooks:
# - guidance 与 xgrammar 在控制粒度上的区别？
# - CFG (Context-Free Grammar) 在 LLM 控制生成中的作用？
# - "类型约束模板"对 schema 演化的友好程度？

import os

USE_MOCK = os.environ.get("USE_REAL_API") != "1"


def build_template():
    # guidance 库通过 CFG 控制生成
    # 安装：pip install guidance
    import guidance

    # 定义带类型约束的模板
    @guidance()
    def user_info(lm, name_desc=""):
        lm += '{{json\n'
        lm += f'  "name": "{name_desc}",\n'
        lm += '"age": {{gen "age" pattern="[0-9]+" stop=","}},\n'
        lm += '"skills": [{{gen "skill" pattern="\\w+" stop=",|\\]"}}]\n'
        lm += '}}\n'
        return lm

    # 调用模型（需要先加载）
    # lm = guidance.models.LlamaCpp("path/to/model.gguf")
    # result = lm + user_info(name_desc="张伟")
    # result["age"]  # 自动是合法整数
    return user_info


if __name__ == "__main__":
    if USE_MOCK:
        print("[mock] guidance 库未启用真实调用。")
        print("\n以下是 guidance 模板的结构示意：")
        template_str = """
@guidance()
def user_info(lm, name_desc=""):
    lm += '{{json\\n'
    lm += f'  "name": "{name_desc}",\\n'
    lm += '"age": {{gen "age" pattern="[0-9]+" stop=","}},\\n'
    lm += '"skills": [{{gen "skill" pattern="\\\\w+" stop=",|\\\\]"}}]\\n'
    lm += '}}\\n'
    return lm
"""
        print(template_str)
        print("[关键点] pattern 强制 age 为数字，skills 元素为单词")
    else:
        tpl = build_template()
        print(f"[Template Created] {tpl}")
