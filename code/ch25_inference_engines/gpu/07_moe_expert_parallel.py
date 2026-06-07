# ---
# chapter: 25
# topic: MoE Expert Parallel (real vLLM config)
# section: 25.5
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: vllm>=0.21.0, torch>=2.5
# run: python 07_moe_expert_parallel.py
# expected_runtime: ~30-60s (model load if available) or fast config-print on missing model
# expected_output: 打印 MoE Expert Parallel 配置 + 解释 EP/TP 通信开销
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.5
# Interview hooks:
#   1. MoE 推理为什么显存省但通信贵？(答: 全部 expert 权重都在显存，token 需 all-to-all)
#   2. Expert Parallel (EP) 和 Tensor Parallel (TP) 的取舍？
#   3. 路由不均衡会怎样？(答: 单 GPU OOM, 可用 expert capacity / drop-and-pad)
#   4. vLLM 0.21.0 启用 EP？(答: enable_expert_parallel=True + 真实 MoE 模型)

"""MoE (Mixture of Experts) Expert Parallel 配置演示 (真实 vLLM 0.21.0).

MoE 把 FFN 切成 N 个 expert, 每个 token 由 router 选 top-k experts.
Expert Parallel (EP) 把不同 expert 放不同 GPU, 通信用 all-to-all.

vLLM 0.21.0 启用 EP:
    LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        tensor_parallel_size=2,
        enable_expert_parallel=True)   # 强制 expert 维度切分

Mixtral-8x7B 约 90GB (fp16) → 单 80GB A100 放不下, 需 EP=2/4 (2×/4× A100-80G)
本脚本:
  1. 检查环境 (多卡 + vllm._C)
  2. 打印 EP 配置 (即使无 MoE 模型也可演示 config)
  3. 可选真跑 Mixtral (有 ≥2 张 80GB 显卡时)

注: Mixtral 在 HF 上 ~90GB;  本地 demo 默认用 Qwen2.5-0.5B 占位 + 仅打印
   config; 真实跑需下载 Mixtral-8x7B-Instruct 权重.
"""
from __future__ import annotations
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu, gpu_summary


# 真实 MoE 模型 (按需取消注释, 默认占位)
# MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # 90GB fp16
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 占位 (非 MoE), 用于演示 config


def main() -> None:
    require_nvidia_gpu(min_vram_gb=16, min_count=1)  # 单卡也跑得起 (config demo)
    print(gpu_summary())
    print()

    # 故意把 import 放在 GPU 检查之后, 避免无 GPU 机器 import vllm._C 失败
    try:
        from vllm import LLM, SamplingParams
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

    # MoE Expert Parallel 配置
    # 注: Mixtral-8x7B 需 ≥2 张 80GB GPU; 此处用 0.5B 占位演示 config
    #     真实 MoE 跑需下载 Mixtral / Qwen2-MoE / DeepSeek-V3 权重
    print("=" * 60)
    print("MoE Expert Parallel 配置演示")
    print("=" * 60)
    print("当前 model 占位: 0.5B (非 MoE, 仅展示 vLLM config)")
    print("真实 MoE 模型 (Mixtral-8x7B 90GB, Qwen2-MoE-A2.7B 28GB):")
    print("  改 MODEL 常量 + enable_expert_parallel=True + tensor_parallel_size≥2")
    print()
    print("提示: vLLM 0.21.0 的 enable_expert_parallel 在 v0.5+ 引入,")
    print("     Mixtral/Qwen2-MoE/DeepSeek-V3 自动启用 MoE 路径.")
    print()

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_num_seqs=4,
        max_model_len=512,
        enable_expert_parallel=True,   # 启用 EP (MoE 模型时生效)
        tensor_parallel_size=1,         # 占位; MoE 真实跑建议 ≥2
        enforce_eager=True,
    )

    # 验证 config 被接受
    vllm_cfg = llm.llm_engine.vllm_config.parallel_config
    print("=" * 60)
    print("vLLM ParallelConfig (MoE 关键字段):")
    print(f"  tensor_parallel_size  = {vllm_cfg.tensor_parallel_size}")
    # enable_expert_parallel 在 vLLM 0.21.0 的 parallel_config 字段
    ep_enabled = getattr(vllm_cfg, "enable_expert_parallel", None)
    print(f"  enable_expert_parallel = {ep_enabled}")
    print(f"  data_parallel_size    = {getattr(vllm_cfg, 'data_parallel_size', 1)}")
    print("=" * 60)
    print()

    # 占位跑一个 prompt (即使非 MoE 模型也跑通, 演示 LLM 链路)
    sampling = SamplingParams(temperature=0.7, max_tokens=32)
    prompts = ["Q: What is MoE?\nA:"]
    outputs = llm.generate(prompts, sampling)
    for i, out in enumerate(outputs):
        text = out.outputs[0].text[:100].replace("\n", " ")
        ctoks = len(out.outputs[0].token_ids)
        print(f"  [{i}] gen={ctoks}t  text={text!r}")
    print()
    print("=" * 60)
    print("MoE Expert Parallel 关键 takeaway:")
    print("  - 全部 expert 权重在显存; 路由不均衡 = 单卡 OOM")
    print("  - EP 通信: all-to-all (NVLink/IB 必需)")
    print("  - 真实 MoE: Mixtral-8x7B (8 expert, top-2) / DeepSeek-V3 (256 expert, top-8)")
    print("  - vLLM 自动处理: prefix cache, chunked prefill, MoE 路由均衡")
    print("=" * 60)


if __name__ == "__main__":
    main()
