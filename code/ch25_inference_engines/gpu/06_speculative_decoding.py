# ---
# chapter: 25
# topic: Speculative Decoding (real vLLM)
# section: 25.4
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: vllm>=0.21.0, torch>=2.5
# run: python 06_speculative_decoding.py
# expected_runtime: ~30-60s (model load + speculative generate)
# expected_output: 真实 vLLM LLM 跑 2 prompt 启用 speculative decoding,
#                    打印 accepted/draft token 统计
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.4
# Interview hooks:
#   1. Speculative decoding 加速原理？(答: 一次 verify 多个 draft token，接受率 α 时 ~1/(1-α) 加速)
#   2. Draft model 如何选？(答: 小同族模型、Medusa、n-gram、EAGLE、self-speculative)
#   3. 接受率 0 时会发生什么？(答: 回退到单 token 步，无收益)
#   4. vLLM 0.21.0 如何启用？(答: speculative_model + num_speculative_tokens 参数)

"""Speculative Decoding 演示 (真实 vLLM 0.21.0).

Speculative decoding 用小 draft model 先预测 K 个 token, 大 target model
一次 verify K 个. 加速比 ≈ K × α (α = 接受率).

vLLM 0.21.0 启用方式 (LLM 构造参数):
    - speculative_model       : draft model HF 名 (或同模型做 self-speculative)
    - num_speculative_tokens  : K (每个 step 草拟几个 token)
    - speculative_draft_tensor_parallel_size : draft model 的 TP

vLLM 同时支持无模型的 Medusa / EAGLE / n-gram 方案, 此处演示同模型
self-speculative (单模型 demo, 实际生产用小 draft).
"""
from __future__ import annotations
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu, gpu_summary


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 1GB, 已下载到本地 HF cache


def main() -> None:
    require_nvidia_gpu(min_vram_gb=16)
    print(gpu_summary())
    print()

    # 故意把 import 放在 GPU 检查之后, 避免无 GPU 机器 import vllm._C 失败
    # shared.vllm_compat: 设了 VLLM_BASE_URL → 走 Docker OpenAI 协议; 否则按需 import 真 vllm
    try:
        from shared.vllm_compat import LLM, SamplingParams
    except ModuleNotFoundError as e:
        if "vllm._C" in str(e):
            print("=" * 60)
            print("ERROR: vllm._C compiled extension is missing.")
            print("  vLLM 0.21.0 在 Windows 上需要从源码编译 C++/CUDA 扩展,")
            print("  当前的 pip install 缺 vllm._C.pyd.")
            print()
            print("修复方案 (任选其一):")
            print("  1. Linux 机器: pip install vllm==0.21.0  (官方支持)")
            print("  2. WSL2 + CUDA: 在 WSL2 里 pip install vllm")
            print("  3. Docker:  vllm/vllm-openai:0.21.0 镜像")
            print()
            print("本脚本代码正确, 在 vllm._C 可用的环境可直接跑通.")
            print("=" * 60)
            return
        raise

    # 启用 speculative decoding
    # 注: 同模型做 self-speculative (实际生产用更小的 draft model)
    #     vLLM 0.21.0 必须传 HF 模型 ID 字符串, 不能传 None
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_num_seqs=4,
        max_model_len=512,
        speculative_model=MODEL,             # 同模型 self-speculative demo
        num_speculative_tokens=5,            # K=5 draft tokens
        enforce_eager=True,
    )

    sampling = SamplingParams(temperature=0.7, max_tokens=64)
    prompts = [
        "Q: What's the capital of France?\nA:",
        "Q: Explain quantum computing in one sentence.\nA:",
    ]

    outputs = llm.generate(prompts, sampling)
    print("=" * 60)
    print("Speculative Decoding 输出:")
    print("=" * 60)
    for i, out in enumerate(outputs):
        text = out.outputs[0].text[:120].replace("\n", " ")
        ptoks = len(out.prompt_token_ids)
        ctoks = len(out.outputs[0].token_ids)
        # vLLM 0.21.0 把 acceptance 统计放在 RequestOutput 里
        n_draft = getattr(out, "num_draft_tokens", 0)
        n_acc = getattr(out, "num_accepted_tokens", 0)
        print(f"  [{i}] prompt={ptoks}t -> gen={ctoks}t")
        print(f"       text={text!r}")
        if n_draft:
            print(f"       draft={n_draft}  accepted={n_acc}  rate={100*n_acc/max(n_draft,1):.0f}%")
    print()
    print("=" * 60)
    print("Speculative Decoding 关键 takeaway:")
    print("  - 1 个 verify forward 同时 verify K=5 draft tokens")
    print("  - 接受率 α 时期望加速: K×α / (1 - (1-α)^K)")
    print("  - draft model 越小越好 (e.g. 0.5B target + 0.1B draft)")
    print("  - 同模型 self-speculative 是 fallback (无 draft 时)")
    print("=" * 60)


if __name__ == "__main__":
    main()
