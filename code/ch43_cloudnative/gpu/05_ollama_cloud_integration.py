# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: cloudnative.ollama_cloud_integration
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: ollama (Python client)
# run: python 05_ollama_cloud_integration.py --mode local
# expected_runtime: depends on the selected local or cloud model
# expected_output: prints only responses returned by the explicitly selected real endpoint
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
# Interview hooks:
#   1. 混合云推理（本地 + Ollama Cloud）的数据合规边界如何划定？
#   2. 用 Bearer Token 调云端 Ollama 时，如何轮转密钥避免被截获？
#   3. ollama.Client 与 openai.OpenAI 客户端的 API 兼容性如何？
"""
本地 Ollama 与 Ollama Cloud 的显式调用示例。

两条路径不能混为一谈：

* 本地服务默认使用 ``http://localhost:11434``；如使用 Ollama 的本地 cloud
  proxy，可把 ``--local-model`` 设为当前 ``ollama list`` 中的 ``*-cloud`` tag。
* 直连 Ollama Cloud 使用官方 host ``https://ollama.com`` 和环境变量
  ``OLLAMA_API_KEY``。云端请求会把 prompt 发往远端，并可能产生费用。

模型清单会变化，应从当前 ``/api/tags`` 或官方模型列表选择，脚本不内置一个
看似永久有效的 cloud model 名。缺依赖、密钥或请求失败时不会回退到 mock。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit local/Ollama Cloud chat demo")
    parser.add_argument("--mode", choices=("local", "cloud", "both"), required=True)
    parser.add_argument(
        "--local-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    parser.add_argument(
        "--local-model",
        default=os.environ.get("OLLAMA_LOCAL_MODEL", "qwen3:8b"),
    )
    parser.add_argument(
        "--cloud-host",
        default=os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com"),
    )
    parser.add_argument(
        "--cloud-model",
        default=os.environ.get("OLLAMA_CLOUD_MODEL", ""),
        help="当前 Ollama Cloud 模型名；必须显式提供或设置 OLLAMA_CLOUD_MODEL",
    )
    parser.add_argument("--prompt", default="Reply with exactly: OK")
    parser.add_argument(
        "--confirm-cloud",
        action="store_true",
        help="确认 prompt 会离开本机且调用可能计费",
    )
    return parser


def _response_content(response: Any) -> str:
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    if isinstance(response, dict):
        content = response.get("message", {}).get("content")
        if isinstance(content, str):
            return content
    raise RuntimeError("Ollama 响应不含 message.content；请核对客户端与服务端版本")


def _load_client_class() -> Any:
    try:
        from ollama import Client
    except ImportError as exc:
        raise RuntimeError("缺少 ollama Python 客户端；请安装当前官方版本。") from exc
    return Client


def _validate_cloud_args(args: argparse.Namespace) -> str:
    if not args.confirm_cloud:
        raise ValueError("cloud/both 模式必须显式添加 --confirm-cloud")
    if urlparse(args.cloud_host).scheme != "https":
        raise ValueError("--cloud-host 必须使用 HTTPS")
    if not args.cloud_model.strip():
        raise ValueError("请用 --cloud-model 或 OLLAMA_CLOUD_MODEL 指定当前可用模型")
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("缺少 OLLAMA_API_KEY")
    return api_key


def _chat(client_class: Any, *, host: str, model: str, prompt: str, headers=None) -> str:
    client = client_class(host=host, headers=headers or {})
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return _response_content(response)


def main() -> int:
    if skip_if_mock("an explicitly selected local or Ollama Cloud endpoint"):
        return 0
    if len(sys.argv) == 1 and os.environ.get("OLLAMA_INTEGRATION_RUN") != "1":
        print(
            "[SKIP] Set OLLAMA_INTEGRATION_RUN=1 and pass --mode only after "
            "reviewing the selected local or cloud endpoint, model, and data boundary."
        )
        print("OK")
        return 0

    args = _parser().parse_args()
    cloud_key = ""
    try:
        if args.mode in ("cloud", "both"):
            cloud_key = _validate_cloud_args(args)
        client_class = _load_client_class()

        if args.mode in ("local", "both"):
            local_reply = _chat(
                client_class,
                host=args.local_host,
                model=args.local_model,
                prompt=args.prompt,
            )
            print("=== Local Ollama ===")
            print(f"Host: {args.local_host}")
            print(f"Model: {args.local_model}")
            print(f"Reply: {local_reply}")

        if args.mode in ("cloud", "both"):
            cloud_reply = _chat(
                client_class,
                host=args.cloud_host,
                model=args.cloud_model,
                prompt=args.prompt,
                headers={"Authorization": f"Bearer {cloud_key}"},
            )
            print("=== Ollama Cloud (remote) ===")
            print(f"Host: {args.cloud_host}")
            print(f"Model: {args.cloud_model}")
            print(f"Reply: {cloud_reply}")
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Ollama request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
