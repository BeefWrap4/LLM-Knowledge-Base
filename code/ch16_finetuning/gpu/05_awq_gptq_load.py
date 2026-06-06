# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.5.2 AWQ / GPTQ 量化模型加载
# difficulty: ⭐⭐⭐
# tier: gpu
# deps: torch, transformers, awq (or auto-gptq)
# run: python 05_awq_gptq_load.py --mock
# expected_runtime: <5s for mock / 30-60s for real (含权重加载)
# expected_output: 演示 AWQ 与 GPTQ 的加载接口差异（mock 模式打印对比表）
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.5.2
# Interview hooks:
#   1. AWQ 与 GPTQ 的核心差异？AWQ 为什么能获得更高精度（激活感知）？
#   2. BitsAndBytes NF4 与 GPTQ/AWQ 的差异？运行时量化 vs 后训练量化？
#   3. fuse_layers=True 的作用？ExLlama / Triton 内核如何加速推理？

"""
AWQ / GPTQ 量化模型加载示例
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_quantization_load():
    """无 GPU / 无权重环境下的对比演示"""
    print("[MOCK] AWQ vs GPTQ vs GGUF vs NF4 对比")
    print()
    print("  方法       类型         原理                  精度   速度")
    print("  --------  ----------  -------------------  -----  ----")
    print("  GPTQ     后训练量化    逐层最小化输出误差     ★★★★   ★★★")
    print("  AWQ      后训练量化    激活感知, 保护重要权重  ★★★★★  ★★★★")
    print("  GGUF     文件格式     llama.cpp 量化         ★★★   ★★★★★")
    print("  NF4      运行时量化    4-bit NormalFloat     ★★★★  即插即用")
    print()
    print("[MOCK] AWQ 加载接口")
    print("  from awq import AutoAWQForCausalLM")
    print("  model = AutoAWQForCausalLM.from_quantized(")
    print("      'TheBloke/Llama-2-7B-AWQ',")
    print("      fuse_layers=True,")
    print("      use_exllama=True,   # ExLlama 内核")
    print("  )")
    print()
    print("[MOCK] GPTQ 加载接口")
    print("  from auto_gptq import AutoGPTQForCausalLM")
    print("  model = AutoGPTQForCausalLM.from_quantized(")
    print("      'TheBloke/Llama-2-7B-GPTQ',")
    print("      device='cuda:0',")
    print("      use_triton=True,    # Triton 加速内核")
    print("  )")
    print()
    print("OK")


def real_quantization_load():
    """真实加载（需 GPU + HuggingFace 权重）"""
    # AWQ 路径
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model = AutoAWQForCausalLM.from_quantized(
            "TheBloke/Llama-2-7B-AWQ",
            fuse_layers=True,
            use_exllama=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")
        print(f"AWQ 模型加载完成: {type(model).__name__}")
    except Exception as e:
        print(f"AWQ 加载失败: {e}")

    # GPTQ 路径
    try:
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            "TheBloke/Llama-2-7B-GPTQ",
            device="cuda:0",
            use_triton=True,
        )
        print(f"GPTQ 模型加载完成: {type(model).__name__}")
    except Exception as e:
        print(f"GPTQ 加载失败: {e}")

    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_quantization_load()
    else:
        real_quantization_load()
