# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.5.4 xgrammar 词表级约束解码
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: xgrammar, transformers, torch
# run: python 18_xgrammar_constrained_decoding.py
# expected_runtime: 5-30s (real)
# expected_output: 打印通过 grammar 约束生成的合法 JSON
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.4
# Interview hooks:
# - 词表级约束能保证什么，不能保证什么？
# - xgrammar 与 outlines/lm-format-enforcer 的实现差异？
# - 为什么约束解码的性能开销必须按后端实测？

import json

try:
    import torch
    import xgrammar as xgr
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    xgr = None
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def run_xgrammar_demo():
    if any(value is None for value in (torch, xgr, AutoConfig, AutoModelForCausalLM, AutoTokenizer)):
        raise RuntimeError("缺少 xgrammar/transformers/torch 可选依赖")
    if not torch.cuda.is_available():
        raise RuntimeError("该 8B 示例需要可用 CUDA GPU")

    # 1. 定义 JSON Schema
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=config.vocab_size
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_json_schema(json.dumps(json_schema))

    prompt = "只输出 JSON：生成一个包含 name、age、skills 的用户信息。"
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    processor = xgr.contrib.hf.LogitsProcessor(compiled)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,
        do_sample=False,
        logits_processor=[processor],
    )

    new_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


if __name__ == "__main__":
    if any(value is None for value in (torch, xgr, AutoConfig, AutoModelForCausalLM, AutoTokenizer)):
        print("[SKIP] 需要 xgrammar、transformers 和 torch")
        print("OK")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("[SKIP] 该 8B 示例需要可用 CUDA GPU")
        print("OK")
        raise SystemExit(0)

    result = run_xgrammar_demo()
    print("[Constrained Output]")
    print(result)
    # 验证是合法 JSON
    parsed = json.loads(result)
    assert isinstance(parsed.get("name"), str)
    assert type(parsed.get("age")) is int and 0 <= parsed["age"] <= 150
    assert set(parsed) <= {"name", "age", "skills"}
    print("[结构与业务校验] 通过")
    print("OK")
