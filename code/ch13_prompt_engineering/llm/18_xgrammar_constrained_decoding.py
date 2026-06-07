# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.4 xgrammar 词表级约束解码
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: xgrammar (可选), transformers (可选), torch (可选)
# run: python 18_xgrammar_constrained_decoding.py
# expected_runtime: <1s (mock) / 5-30s (real)
# expected_output: 打印通过 grammar 约束生成的合法 JSON
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.4
# Interview hooks:
# - 词表级约束如何保证 100% 输出合法 JSON？
# - xgrammar 与 outlines/lm-format-enforcer 的实现差异？
# - 约束解码对推理速度的开销 (5-15%) 主要来自哪里？

import os
import json

USE_MOCK = os.environ.get("USE_REAL_API") != "1"


def run_xgrammar_demo():
    # xgrammar：2025 年发布的开源结构化生成引擎
    # 安装：pip install xgrammar
    if USE_MOCK:
        # 演示约束效果：返回一个 schema-合规的 JSON
        return json.dumps({
            "name": "张伟",
            "age": 28,
            "skills": ["Python", "Rust"]
        }, ensure_ascii=False)

    import xgrammar as xg
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. 定义 JSON Schema
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "skills": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    }

    # 2. 编译为 Grammar
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    grammar = xg.Grammar.from_json_schema(json_schema)
    compiler = xg.GrammarCompiler(tokenizer)
    compiled_grammar = compiler.compile_grammar(grammar)

    # 3. 在推理时强制约束
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16
    ).cuda()

    input_ids = tokenizer.encode("请生成一个用户信息：")
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=False,
        compiled_grammar=compiled_grammar,  # 关键：传入编译后的 grammar
    )

    # 输出保证是合法 JSON
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    result = run_xgrammar_demo()
    print("[Constrained Output]")
    print(result)
    # 验证是合法 JSON
    parsed = json.loads(result)
    assert "name" in parsed and "age" in parsed
    print("[Schema 验证] 通过")
