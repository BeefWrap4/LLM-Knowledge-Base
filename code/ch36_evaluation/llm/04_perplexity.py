# ---
# chapter: 36
# topic: 大模型评估基础
# topic_id: evaluation.perplexity
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: torch, transformers
# run: python 04_perplexity.py
# expected_runtime: <2s (mock mode) / model-dependent (real)
# expected_output: Measured PPL values for two inputs in explicit real mode
# ---
# See: ../../../36_大模型评估基础.md
# Interview hooks:
# - How is Perplexity related to Cross-Entropy Loss?
# - Why is Perplexity not directly comparable across different tokenizers?
# - Can Perplexity measure factual correctness? Why or why not?

"""Perplexity 计算示例（使用 Hugging Face Transformers）。

困惑度衡量语言模型对给定文本的"意外程度"。困惑度越低，模型对文本预测越好。
默认 ``LLM_MOCK=1`` 跳过模型加载；``LLM_MOCK=0`` 才允许加载本地缓存或联网下载。
"""

import os


def compute_perplexity(
    text: str,
    model_name: str | None = None,
) -> float:
    if os.environ.get("LLM_MOCK", "1") != "0":
        raise RuntimeError("LLM_MOCK=1：未加载模型；真实计算请显式设置 LLM_MOCK=0")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("真实模式需要 torch 和 transformers") from exc

    resolved_model = model_name or os.environ.get(
        "PPL_MODEL",
        "openai-community/gpt2",
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    model = AutoModelForCausalLM.from_pretrained(resolved_model)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    max_len = model.config.max_position_embeddings
    input_ids = encodings.input_ids[:, :max_len]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-Entropy Loss

    ppl = torch.exp(loss).item()
    return ppl


if __name__ == "__main__":
    if os.environ.get("LLM_MOCK", "1") != "0":
        print("[SKIP] LLM_MOCK=1：未加载语言模型，未生成伪 PPL")
    else:
        text_fluent = "The weather is beautiful today and I plan to go for a walk."
        text_gibberish = "The weather beautiful today plan walk for go and I a to."
        ppl_fluent = compute_perplexity(text_fluent)
        ppl_gibberish = compute_perplexity(text_gibberish)

        print(f"流畅文本 PPL: {ppl_fluent:.2f}")
        print(f"乱序文本 PPL: {ppl_gibberish:.2f}")
    print("OK")
