# ---
# chapter: 40
# topic: 推理内存、量化与批处理
# topic_id: inference_engines.speculative_decoding
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: vllm>=0.21.0, torch>=2.5
# run: python 06_speculative_decoding.py
# expected_runtime: ~30-60s (model load + speculative generate)
# expected_output: 真实 vLLM LLM 跑 2 prompt，启用 n-gram speculative decoding
# ---
# See: ../../../40_推理内存量化与批处理.md
# Interview hooks:
#   1. Speculative decoding 加速原理？(答: proposer 草拟多个 token，target 并行验证；收益需实测)
#   2. Draft model 如何选？(答: 小同族模型、Medusa、n-gram、EAGLE、self-speculative)
#   3. 接受率 0 时会发生什么？(答: 回退到单 token 步，无收益)
#   4. vLLM 0.21.0 如何启用？(答: LLM(..., speculative_config={...}))

"""Speculative Decoding 演示 (真实 vLLM 0.21.0).

Speculative decoding 由 proposer 草拟多个 token，再由 target model 并行验证。
实际收益取决于接受率、proposer 成本、流量、硬件和采样配置，不能只用 K 或 α 推出。

vLLM 0.21.0 启用方式 (LLM 构造参数):
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 5,
    }

vLLM 还支持 draft model、EAGLE、MTP、suffix 等方案；本例用无需额外权重的 n-gram。
"""

from __future__ import annotations

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import (
    gpu_summary,
    require_nvidia_gpu,
    skip_if_mock,
    skip_unless_enabled,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 1GB, 已下载到本地 HF cache


def main() -> None:
    if skip_if_mock("Linux、NVIDIA GPU、vLLM 编译扩展和本地模型"):
        return
    if skip_unless_enabled(
        "VLLM_EXAMPLE_RUN", "the Linux/WSL2 vLLM runtime and local model weights"
    ):
        return
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

    # vLLM 0.21.0 的公共入口是 speculative_config。
    # n-gram proposer 不需要额外 draft 权重，适合演示配置和结果等价性。
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_num_seqs=4,
        max_model_len=512,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        },
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
        print(f"  [{i}] prompt={ptoks}t -> gen={ctoks}t")
        print(f"       text={text!r}")
    print()
    print("=" * 60)
    print("Speculative Decoding 关键 takeaway:")
    print("  - proposer 草拟 token，target model 并行验证")
    print("  - 在相同采样配置下保持 target 分布，不以近似输出换速度")
    print("  - 本例使用 n-gram proposer；draft model、EAGLE、MTP 需分别配置")
    print("  - 必须用目标流量实测 TTFT、TPOT、吞吐、显存和接受率")
    print("=" * 60)


if __name__ == "__main__":
    main()
