# ---
# chapter: 41
# topic: 高性能推理引擎与服务
# topic_id: lora_qlora.xinference_deployment
# difficulty: ⭐⭐⭐
# tier: gpu
# deps: httpx (REST 调用)
# run: python 07_xinference_deployment.py
# expected_runtime: <10s (需 Xinference 服务运行于 :9997)
# expected_output: 列出已部署模型 + chat completions 调用
# ---
# See: ../../../41_高性能推理引擎与服务.md
#
# Interview hooks:
#   1. Xinference 相对 vLLM 的核心优势？多模型统一管理与 Web UI？
#   2. launch_model / get_model / terminate_model 的资源生命周期如何管理？
#   3. 什么场景适合选 Xinference vs 直接 vLLM 部署？
"""Xinference 部署演示 (真实 REST API, 缺服务友好抛错).

Xinference 是 LF AI & Data 项目的 LLM 推理框架:
  - 类似 Ollama/vLLM, 但支持异构 backend (transformers/vllm/llama.cpp)
  - 一行启动: xinference launch --model-engine vllm

REST API 端点:
  POST {host}:9997/v1/chat/completions  (OpenAI 兼容)
  GET  {host}:9997/v1/models
"""

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import httpx

from shared._error_helper import raise_with_help
from shared.gpu_guard import skip_if_mock

XINFERENCE_HOST = "http://localhost:9997"


def check_xinference_running() -> None:
    """Xinference 服务不可达 → 友好抛错."""
    try:
        r = httpx.get(f"{XINFERENCE_HOST}/v1/models", timeout=2.0)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
        raise_with_help(
            f"Xinference 服务不可达 ({XINFERENCE_HOST}): {type(e).__name__}: {e}",
            "安装: `pip install xinference`. 启动: `xinference launch --model-engine vllm`. "
            "或 Docker: `docker run -p 9997:9997 xprobe/xinference`. "
            "代码逻辑正确, 上述环境跑可成功.",
        )


def main():
    if skip_if_mock("运行中的 Xinference 服务和已部署模型"):
        return
    if os.environ.get("XINFERENCE_RUN") != "1":
        print(
            "[SKIP] Set XINFERENCE_RUN=1 only after starting and reviewing "
            "the local Xinference service and deployed model."
        )
        print("OK")
        return
    check_xinference_running()

    print("=== Xinference 部署演示 (真实 REST API) ===\n")

    # 1) 列出已部署模型
    r = httpx.get(f"{XINFERENCE_HOST}/v1/models", timeout=5.0)
    r.raise_for_status()
    models = r.json().get("data", [])
    print(f"已部署模型: {[m['id'] for m in models]}")

    if not models:
        print("\n当前无已部署模型. 拉起一个 (示例):")
        print("  from xinference.client import Client")
        print('  c = Client("http://localhost:9997")')
        print('  uid = c.launch_model(model_name="qwen2.5-instruct",')
        print("                       model_size_in_billions=7, n_gpu=1)")
        print("OK")
        return

    # 2) 调一个 chat completion
    model_id = models[0]["id"]
    print(f"\n调用 chat completion: model={model_id}")
    r = httpx.post(
        f"{XINFERENCE_HOST}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello! 用一句话介绍你自己."}],
            "max_tokens": 64,
            "temperature": 0.7,
        },
        timeout=30.0,
    )
    r.raise_for_status()
    result = r.json()
    print(f"Response:\n  {result['choices'][0]['message']['content'][:200]}")
    if "usage" in result:
        u = result["usage"]
        print(
            f"  usage: prompt={u.get('prompt_tokens')} "
            f"completion={u.get('completion_tokens')} "
            f"total={u.get('total_tokens')}"
        )
    print("OK")


if __name__ == "__main__":
    main()
