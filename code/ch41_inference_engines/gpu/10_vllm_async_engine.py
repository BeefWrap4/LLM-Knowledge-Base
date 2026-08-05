# ---
# chapter: 41
# topic: 高性能推理引擎与服务
# topic_id: inference_engines.vllm_async_engine
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: vllm
# run: python 10_vllm_async_engine.py
# expected_runtime: 30-120s (model load + 5 stream requests)
# expected_output: 流式生成 5 个 prompt 的响应
# ---
# See: ../../../41_高性能推理引擎与服务.md
# Interview hooks:
#   1. vLLM 0.x → 0.4+ 的 API 变化？(答: LLMEngine → AsyncLLMEngine, 同步→async stream)
#   2. SamplingParams 的关键字段？(答: temperature, top_p, top_k, max_tokens, stop)
#   3. vLLM 如何做 OpenAI 兼容 server？(答: --served-model-name + FastAPI 包装)
"""vLLM AsyncLLMEngine 流式生成演示 (真实 vLLM).

AsyncLLMEngine 是 vLLM 0.4+ 的核心: 单进程多请求流式 batched 生成.
相对同步逐请求执行，continuous batching 可改善资源利用率；幅度取决于模型、
输入/输出长度、并发、硬件与延迟 SLO。
"""

import asyncio
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu, skip_if_mock, skip_unless_enabled


def check_hardware():
    require_nvidia_gpu(min_vram_gb=24)


def check_vllm_engine():
    """检查 vllm._C 编译扩展 (Windows 不可用)."""
    # shared.vllm_compat: 设了 VLLM_BASE_URL → 走 Docker OpenAI 协议; 否则按需 import 真 vllm
    try:
        from shared.vllm_compat import AsyncLLMEngine, AsyncEngineArgs  # noqa
        import vllm._C  # noqa
    except (ImportError, ModuleNotFoundError) as e:
        raise_with_help(
            f"vllm._C 编译扩展不可用: {e}",
            "vLLM 0.21.0 在 Windows 上官方不支持 (无 wheel). "
            "修复路径: 1) Linux + pip install vllm; 2) WSL2; "
            "3) Docker vllm/vllm-openai:0.21.0. "
            "代码逻辑正确, 上述环境跑可成功.",
        )


async def main():
    if skip_if_mock("an NVIDIA GPU, CUDA, vLLM, and local model weights"):
        return
    if skip_unless_enabled(
        "VLLM_EXAMPLE_RUN", "the Linux/WSL2 vLLM runtime and local model weights"
    ):
        return
    check_hardware()
    check_vllm_engine()

    from shared.vllm_compat import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    # 加载模型 (0.5B 已有, 7B 需下)
    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default` (1.7GB) 或 `make download-models-llm` (15GB for 7B).",
        )

    args = AsyncEngineArgs(
        model=model_path,
        max_num_seqs=8,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        enforce_eager=True,  # 避免 CUDA graph 编译开销
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    sampling = SamplingParams(temperature=0.7, max_tokens=64)

    # 流式生成 5 个并发请求
    prompts = [
        "Q: 什么是大语言模型? A:",
        "Q: 解释 Python GIL. A:",
        "Q: vLLM 是什么? A:",
        "Q: 什么是 GPU? A:",
        "Q: 介绍 transformer. A:",
    ]

    async def generate_one(req_id: str, prompt: str) -> str:
        """流式生成单个请求, 收集完整响应."""
        full_text = ""
        async for out in engine.generate(prompt, sampling, req_id):
            if out.finished:
                full_text = out.outputs[0].text
            else:
                # 实时打印 streaming
                new_text = out.outputs[0].text[len(full_text) :]
                if new_text:
                    print(f"[{req_id}] {new_text}", end="", flush=True)
                    full_text = out.outputs[0].text
        print()  # 换行
        return full_text

    # 并发生成
    print("=== 流式生成 5 个并发请求 (max_tokens=64) ===\n")
    tasks = [generate_one(f"r{i}", p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    print("\n=== 完成 ===")
    print(f"  总生成 tokens: ~{sum(len(r.split()) for r in results)}")
    print(f"  并发请求数: {len(results)}")
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
