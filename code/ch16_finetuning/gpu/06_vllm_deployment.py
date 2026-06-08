# ---
# chapter: 16
# topic: vLLM 部署 (真实 vLLM, 缺 vllm._C 友好抛错)
# section: 16.6.2
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: vllm (Linux only), torch
# run: python 06_vllm_deployment.py
# expected_runtime: <5s (脚本模式) / serve 模式长跑
# expected_output: 真实 vLLM LLM 类推理 Qwen2.5-0.5B-Instruct
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.6.2
#
# Interview hooks:
#   1. vLLM 的 PagedAttention + Continuous Batching 如何实现高吞吐？
#   2. gpu_memory_utilization=0.85 的含义？为什么不能设到 1.0？
#   3. tensor_parallel_size 与 Pipeline Parallelism 的使用场景差异？
"""vLLM 部署演示 (真实 vLLM 引擎, Windows 友好抛错).

生产部署:
    vllm serve Qwen2.5-0.5B-Instruct --port 8000 --tensor-parallel-size 1

OpenAI 客户端:
    from openai import OpenAI
    client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
    resp = client.chat.completions.create(
        model='Qwen2.5-0.5B-Instruct',
        messages=[{'role': 'user', 'content': 'Hello!'}])
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


def check_vllm_engine():
    """Windows 上 vllm._C 缺失 → 友好抛错."""
    try:
        import vllm._C  # noqa: F401
        from vllm import LLM  # noqa: F401
    except (ImportError, ModuleNotFoundError) as e:
        raise_with_help(
            f"vllm._C 不可用: {e}",
            "vLLM 0.21.x 在 Windows 上官方不支持 (C++ 扩展未编译). "
            "修复: (1) Linux + `pip install vllm`; (2) WSL2; "
            "(3) Docker: `docker run --gpus all -p 8000:8000 vllm/vllm-openai:0.21.0`. "
            "代码逻辑正确, 上述环境跑可成功.",
        )


def main():
    check_hardware()
    check_vllm_engine()

    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default` 下载 Qwen2.5-0.5B-Instruct.",
        )

    from vllm import LLM, SamplingParams

    print("=== vLLM 部署演示 (真实 vLLM) ===\n")
    print("生产部署命令:")
    print(f"  vllm serve {model_path} \\")
    print("    --port 8000 --tensor-parallel-size 1 --max-model-len 4096")
    print()
    print("OpenAI 客户端调用:")
    print("  from openai import OpenAI")
    print("  client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')")
    print("  resp = client.chat.completions.create(")
    print("      model='Qwen2.5-0.5B-Instruct',")
    print("      messages=[{'role': 'user', 'content': 'Hello!'}])")
    print()

    # 直接 LLM 类 (等价 serve 模式, 单次推理演示)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        enforce_eager=True,  # 跳过 CUDA graph 编译, 启动更快
        dtype="bfloat16",
    )
    sampling = SamplingParams(temperature=0.7, max_tokens=32)
    outputs = llm.generate(["Hello! Introduce yourself briefly."], sampling)
    print(f"Direct LLM result:\n  {outputs[0].outputs[0].text[:200]}")


if __name__ == "__main__":
    main()
