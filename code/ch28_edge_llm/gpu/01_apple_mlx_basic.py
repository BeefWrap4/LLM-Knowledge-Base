# ---
# chapter: 28
# topic: Apple MLX 基础推理 (真实 MLX)
# section: 28.3 Apple MLX 框架
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: mlx, mlx-lm
# run: python 01_apple_mlx_basic.py
# expected_runtime: 1-3s (loading) + ~5-10s/tok (推理) on Apple Silicon
# expected_output: 真实 MLX 模型加载与 token 生成
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.3
# Interview hooks:
#   1. Apple MLX 与 PyTorch MPS 核心区别是什么?
#   2. MLX 为什么特别适合 Apple Silicon 端侧 LLM 推理?
#   3. mlx-community 模型仓库的 Q4 量化模型有什么优势?
"""Apple MLX 基础推理 (真实 mlx_lm 调用, 需 Apple Silicon)."""

from __future__ import annotations

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_apple_silicon, skip_if_mock, skip_unless_apple_silicon


# === 硬件检查函数 (供测试用) ===
def check_hardware() -> None:
    """调用 require_apple_silicon() 抛友好错 (非 Apple Silicon 时)."""
    require_apple_silicon()


# === 主代码 ===
def main() -> None:
    if skip_if_mock("Apple Silicon、MLX 依赖和本地模型权重"):
        return
    if skip_unless_apple_silicon("Apple Silicon、MLX 依赖和本地模型权重"):
        return
    check_hardware()

    # 1. 真实 MLX 调用: 加载 4-bit 量化模型并生成
    try:
        from mlx_lm import generate, load  # noqa: PLC0415
    except ImportError as e:
        raise_with_help(
            f"无法 import mlx_lm: {e}",
            "在 Apple Silicon Mac 上运行 `pip install mlx mlx-lm` 安装.",
        )

    model_path = str(_code_root / "models" / "Qwen2.5-7B-Instruct-4bit-mlx")
    if not Path(model_path).exists():
        raise_with_help(
            f"找不到 MLX 模型 {model_path}",
            "运行 `make download-models-edge` 下载 MLX 4-bit 量化模型, "
            "或手动从 https://huggingface.co/mlx-community 下载 "
            "Qwen2.5-7B-Instruct-4bit 子目录到 code/models/.",
        )

    print(f"加载模型: {model_path}")
    model, tokenizer = load(model_path)

    prompt = "用中文讲一个关于 Apple MLX 统一内存的笑话"
    print(f"Prompt: {prompt}")

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=128,
        verbose=True,
    )
    print(f"\nMLX response: {response}")
    print("OK")


if __name__ == "__main__":
    main()
