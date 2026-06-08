# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.2 Diffusion LLM - SGLang 部署 LLaDA 2.0
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: sglang (真实模式)
# run: python 09_sglang_diffusion_llm.py
# expected_runtime: <5s (mock)
# expected_output: Diffusion LLM 推理 prompt 演示
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-2-diffusion-llm：扩散语言模型
# Interview hooks:
#   1. Mask Diffusion 与 AR 自回归生成的核心差异？
#   2. Diffusion LLM 的 KV Cache 复用如何实现？
#   3. 为什么 Diffusion LLM 对代码生成特别有优势？

import os


def main():
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    if use_mock:
        # 演示 Diffusion LLM 推理时的关键参数
        prompt = "用Python实现快速排序，并解释复杂度。\n\n"
        params = {
            "max_tokens": 512,
            "diffusion_steps": 16,
            "temperature": 0.7,
        }
        print("Diffusion LLM (mock) inference:")
        print(f"  prompt: {prompt.strip()}")
        print(f"  params: {params}")
        print("Diffusion LLM demo OK")
        return

    # 真实模式：使用 SGLang 部署 LLaDA
    try:
        import sglang as sgl
    except ImportError:
        print("sglang not installed. Skipping real mode.")
        return

    @sgl.function
    def diffusion_complete(s, prompt):
        s += prompt
        s += sgl.gen("answer", max_tokens=512, diffusion_steps=16, temperature=0.7)

    runtime = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(runtime)
    state = diffusion_complete.run(prompt="用Python实现快速排序，并解释复杂度。\n\n")
    print(state["answer"])


if __name__ == "__main__":
    main()
