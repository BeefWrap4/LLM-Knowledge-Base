# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.2.4 Perplexity
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: torch, transformers
# run: python 04_perplexity.py
# expected_runtime: 30-60s (first run downloads gpt2)
# expected_output: Fluent text PPL much lower than gibberish text PPL
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - How is Perplexity related to Cross-Entropy Loss?
# - Why is Perplexity unreliable for comparing models of different sizes?
# - Can Perplexity measure factual correctness? Why or why not?

"""Perplexity 计算示例（使用 Hugging Face Transformers）。

困惑度衡量语言模型对给定文本的"意外程度"。困惑度越低，模型对文本预测越好。
"""


def compute_perplexity(text: str, model_name: str = "gpt2") -> float:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[mock] transformers/torch 未安装。返回 mock PPL=100.0")
        return 100.0

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        # 网络/HF 离线时返回 mock PPL
        print(f"[mock] 模型 {model_name} 加载失败 ({type(e).__name__})，返回 mock PPL=100.0")
        return 100.0

    encodings = tokenizer(text, return_tensors="pt")
    max_len = model.config.max_position_embeddings
    input_ids = encodings.input_ids[:, :max_len]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-Entropy Loss

    ppl = torch.exp(loss).item()
    return ppl


if __name__ == "__main__":
    # 测试
    text_fluent = "The weather is beautiful today and I plan to go for a walk."
    text_gibberish = "The weather beautiful today plan walk for go and I a to."

    try:
        ppl_fluent = compute_perplexity(text_fluent)
        ppl_gibberish = compute_perplexity(text_gibberish)

        print(f"流畅文本 PPL: {ppl_fluent:.2f}")
        print(f"乱序文本 PPL: {ppl_gibberish:.2f}")
        # 流畅文本的 PPL 显著低于乱序文本
    except Exception as exc:
        print(f"[mock] 计算失败 ({exc})，使用模拟值")
        print("流畅文本 PPL: 35.20")
        print("乱序文本 PPL: 280.50")
