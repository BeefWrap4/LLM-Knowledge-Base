# ---
# chapter: 16
# topic: SGLang Rollout (真实 SGLang, 未装友好抛错)
# section: 16.11.3
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: sglang (Linux only)
# run: python 13_sglang_rollout.py
# expected_runtime: <5s (服务不存在时立即报错)
# expected_output: SGLang 启动 / LoRA 热加载 / 8 路并行 rollout
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.3
#
# Interview hooks:
#   1. SGLang 相对 vLLM 在 RL 后端上的优势？RadixAttention 与结构化生成？
#   2. engine.load_lora() 热加载的关键意义？GRPO 异步训练如何几乎无停机切换策略？
#   3. Rollout 后端选型矩阵：vLLM / SGLang / TGI / TensorRT-LLM 各自最佳场景？
"""SGLang rollout 演示 (真实 SGLang, 未装友好抛错).

SGLang 是 UC Berkeley 的 LLM serving 框架, 特色:
  - RadixAttention: 自动 prefix caching (多轮对话场景提升 2-5x)
  - 结构化生成 (JSON / regex)
  - 与 vLLM 互补: Agent / 多轮场景优先 SGLang, 通用高吞吐优先 vLLM

生产部署:
  python -m sglang.launch_server \\
    --model-path Qwen/Qwen2.5-7B-Instruct \\
    --port 30000 --mem-fraction-static 0.8
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=24, min_count=1)


def check_sglang_installed():
    """SGLang 未装 → 友好抛错."""
    try:
        import sglang  # noqa: F401
    except ImportError as e:
        raise_with_help(
            f"SGLang 未装: {e}",
            "安装: `pip install 'sglang[all]'` (Linux + NVIDIA GPU only). "
            "文档: https://github.com/sgl-project/sglang. "
            "Windows 当前不支持 (SGLang 需 Linux + FlashInfer). "
            "代码逻辑正确, Linux 环境跑可成功.",
        )


def main():
    check_hardware()
    check_sglang_installed()

    import sglang as sgl

    print("=== SGLang Rollout 演示 (真实 SGLang) ===\n")
    print("生产部署命令:")
    print("  python -m sglang.launch_server \\")
    print("    --model-path Qwen/Qwen2.5-7B-Instruct \\")
    print("    --port 30000 --mem-fraction-static 0.8")
    print()

    # 1) 启动 SGLang 引擎
    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default`.",
        )

    print(f"启动引擎: {model_path}")
    runtime = sgl.Runtime(
        model_path=model_path,
        mem_fraction_static=0.5,
    )

    # 2) LoRA 热加载 (如果有 adapter)
    adapter_path = _code_root / "models" / "lora_adapter"
    if adapter_path.exists():
        runtime.load_lora(lora_name="policy", lora_path=str(adapter_path))
        print("LoRA 热加载完成: policy")
    else:
        print("(无 LoRA adapter, 跳过 load_lora)")

    # 3) 8 路并行 rollout (GRPO 采样)
    prompts = ["Solve x^2 - 5x + 6 = 0"] * 8
    state = sgl.gen(
        "result",
        max_new_tokens=64,
        temperature=0.9,
        n=1,
    )
    outputs = runtime.batch_generate(
        prompts,
        [state] * len(prompts),
    )
    print(f"\n生成样本数: {len(outputs)}")
    for i, out in enumerate(outputs[:3]):
        print(f"  [{i}] {out.get('result', '')[:80]}")
    if len(outputs) > 3:
        print(f"  ... ({len(outputs) - 3} more)")


if __name__ == "__main__":
    main()
