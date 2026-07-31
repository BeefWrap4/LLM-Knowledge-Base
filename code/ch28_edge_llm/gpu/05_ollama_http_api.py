# ---
# chapter: 28
# topic: Ollama OpenAI 兼容 API 调用 (真实 httpx / openai SDK)
# section: 28.4.2 Ollama 一键部署
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: httpx, openai
# run: python 05_ollama_http_api.py
# expected_runtime: <30s (受 Ollama 服务可用性 + 模型下载状态影响)
# expected_output: 真实调用 Ollama /api/chat 与 /v1/chat/completions 拿到回复
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.4.2
# Interview hooks:
#   1. Ollama 与 llama.cpp 直接调用的关系是什么?
#   2. Ollama 的 OpenAI 兼容端点路径是什么?
#   3. 如何在 Python 中以 OpenAI SDK 客户端调用本地 Ollama?
"""Ollama 本地 LLM 服务的两种 API 调用方式 (真实调用, 非打印代码字符串)."""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本既能 `python file.py` 也能 `import` 找到 shared/
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_ollama, skip_if_mock, skip_unless_enabled


def call_ollama_native(prompt: str = "Hello!", model: str = "llama3.2:3b") -> str:
    """Ollama 原生 API: POST /api/chat."""
    import httpx

    try:
        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except httpx.ConnectError:
        raise_with_help(
            "Ollama 服务未运行 (connection refused on :11434).",
            "先 `ollama serve` (后台启动); 然后 `ollama pull llama3.2:3b`.",
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise_with_help(
                f"模型 {model} 未下载 (HTTP 404 from /api/chat).",
                f"运行 `ollama pull {model}`.",
            )
        raise


def call_ollama_openai_compat(prompt: str = "Hello!", model: str = "llama3.2:3b") -> str:
    """Ollama OpenAI 兼容端点: POST /v1/chat/completions (可用 openai SDK)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise_with_help("openai SDK 未装.", "运行 `make install-llm` 或 `pip install openai`.")

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise_with_help(
            f"Ollama OpenAI 兼容端点调用失败: {type(e).__name__}: {e}",
            "确保 `ollama serve` 已运行, 且 `ollama pull llama3.2:3b` 已下载模型.",
        )


def main() -> None:
    if skip_if_mock("运行中的 Ollama 服务和已下载的 llama3.2:3b 模型"):
        return
    if skip_unless_enabled(
        "OLLAMA_EXAMPLE_RUN", "the local Ollama service, installed model, and prompt boundary"
    ):
        return
    # 1) 健康检查: Ollama 服务 + 模型可用性
    require_ollama(model="llama3.2:3b")

    print("=== Ollama 原生 API ===")
    result = call_ollama_native("用一句话介绍你自己, 30 字以内.")
    print(f"Response: {result}\n")

    print("=== Ollama OpenAI 兼容 ===")
    result = call_ollama_openai_compat("用一句话介绍你自己, 30 字以内.")
    print(f"Response: {result}")
    print("OK")


if __name__ == "__main__":
    main()
