# ---
# chapter: 28
# topic: Apple MLX 基础推理
# section: 28.3 Apple MLX 框架
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: mlx, mlx-lm
# run: python 01_apple_mlx_basic.py
# expected_runtime: <1s (mock mode)
# expected_output: 模拟 MLX 模型加载和生成 token
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.3
# Interview hooks:
#   1. Apple MLX 与 PyTorch MPS 核心区别是什么?
#   2. MLX 为什么特别适合 Apple Silicon 端侧 LLM 推理?
#   3. mlx-community 模型仓库的 Q4 量化模型有什么优势?
"""Apple MLX 基础推理 (mock 模式 - 不需要 Apple Silicon 也能跑)."""
from __future__ import annotations

import os
import sys

# 是否在 Apple Silicon 上
ON_APPLE_SILICON = sys.platform == "darwin"


def mock_mlx_generate(prompt: str, max_tokens: int = 32) -> str:
    """模拟 MLX 模型推理. 真实环境会调用 mlx_lm.load/generate."""
    # 真实代码:
    # import mlx.core as mx
    # from mlx_lm import load, generate
    # model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    # text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)
    return f"[MLX-mock] generated {max_tokens} tokens for prompt: {prompt[:30]!r}"


def detect_mlx_environment() -> dict:
    """探测运行环境, 用于决定是否能用真 MLX."""
    info = {
        "platform": sys.platform,
        "apple_silicon": ON_APPLE_SILICON,
        "mlx_available": False,
    }
    try:
        import mlx.core  # noqa: F401
        info["mlx_available"] = True
    except ImportError:
        pass
    return info


def main() -> None:
    env = detect_mlx_environment()
    print(f"运行环境: {env}")

    if not env["apple_silicon"]:
        print("⚠️  非 Apple Silicon 平台, 使用 mock 推理")
    elif not env["mlx_available"]:
        print("⚠️  未安装 mlx-lm, 使用 mock 推理 (pip install mlx mlx-lm)")
    else:
        print("✅ 检测到 MLX, 可加载 mlx-community 模型")

    # 模拟推理
    prompt = "解释 Apple MLX 的统一内存架构"
    result = mock_mlx_generate(prompt, max_tokens=64)
    print(result)

    # 真实 MLX 调用模板 (注释掉, 需要 Apple Silicon + 4GB 量化模型)
    # from mlx_lm import load, generate
    # model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    # text = generate(model, tokenizer, prompt=prompt, max_tokens=100)


if __name__ == "__main__":
    main()
    print("OK")
