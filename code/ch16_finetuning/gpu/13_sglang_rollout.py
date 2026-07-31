# ---
# chapter: 16
# topic: SGLang Rollout (OpenAI-compatible local service)
# section: 16.10.3
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib client; SGLang server is an external prerequisite)
# run: python 13_sglang_rollout.py
# expected_runtime: <10s after a local server is ready
# expected_output: 8 concurrent rollout responses
# ---
# See: ../../../16_模型微调与推理优化.md §16.10.3
#
# Interview hooks:
#   1. 为什么训练框架、rollout backend 和 API client 应分层？
#   2. 如何验证策略权重版本、采样参数和请求结果之间的一致性？
#   3. 如何用相同流量比较 SGLang、vLLM 与框架内置 rollout？
"""通过本机 SGLang 的 OpenAI-compatible API 并发采样。

先按官方文档启动服务，例如：

  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000

本脚本不调用易随版本变化的进程内 Runtime/LoRA API，也不声称 SGLang 在所有负载上优于
其他后端。默认 mock 模式跳过；真实路径只允许回环地址，避免误向外部端点发送提示词。
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock


def validate_loopback_base_url(base_url: str) -> str:
    """只接受本机 HTTP(S) 服务，并返回无尾斜杠 URL。"""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("SGLANG_BASE_URL 必须使用 http 或 https")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("教学脚本只允许连接本机 SGLang；请使用回环地址")
    return base_url.rstrip("/")


def request_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    """向本机服务发送 JSON 请求；API key 只从环境读取且不会输出。"""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("SGLANG_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{base_url}{path}", data=body, headers=headers)
    timeout = float(os.getenv("SGLANG_TIMEOUT_SECONDS", "30"))
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is loopback-validated
        return json.loads(response.read().decode("utf-8"))


def discover_model(base_url: str) -> str:
    """优先使用显式模型 ID，否则读取 OpenAI-compatible /v1/models。"""
    configured = os.getenv("SGLANG_MODEL", "").strip()
    if configured:
        return configured
    result = request_json(base_url, "/v1/models")
    models = result.get("data", [])
    if not models or not isinstance(models[0].get("id"), str):
        raise RuntimeError("SGLang /v1/models 未返回可用模型；可设置 SGLANG_MODEL")
    return models[0]["id"]


def sample_once(base_url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 64,
    }
    result = request_json(base_url, "/v1/chat/completions", payload)
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("SGLang 响应缺少 choices")
    return str(choices[0].get("message", {}).get("content", ""))


def main() -> None:
    if skip_if_mock("本机已启动的 SGLang 服务和对应 GPU/模型"):
        return
    if os.getenv("SGLANG_ROLLOUT_RUN") != "1":
        print(
            "[SKIP] Set SGLANG_ROLLOUT_RUN=1 only after starting and reviewing "
            "the loopback SGLang service and served model."
        )
        print("OK")
        return

    base_url = validate_loopback_base_url(
        os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
    )
    model = discover_model(base_url)
    prompt = "Solve x^2 - 5x + 6 = 0 and give only the final roots."

    print("=== SGLang local rollout client ===")
    print(f"endpoint: {base_url}")
    print(f"model: {model}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda _: sample_once(base_url, model, prompt), range(8)))

    print(f"rollouts: {len(outputs)}")
    for index, output in enumerate(outputs[:3]):
        print(f"  [{index}] {output[:120]}")
    print("OK")


if __name__ == "__main__":
    main()
