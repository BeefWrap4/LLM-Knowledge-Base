# ---
# chapter: 28
# topic: Ollama OpenAI 兼容 API 调用
# section: 28.4.2 Ollama 一键部署
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: httpx (or urllib stdlib)
# run: python 05_ollama_http_api.py
# expected_runtime: <1s (no live server, prints curl/python equivalent)
# expected_output: Ollama /api/chat 与 /v1/chat/completions 调用示例
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.4.2
# Interview hooks:
#   1. Ollama 与 llama.cpp 直接调用的关系是什么?
#   2. Ollama 的 OpenAI 兼容端点路径是什么?
#   3. 如何在 Python 中以 OpenAI SDK 客户端调用本地 Ollama?
"""Ollama 本地 LLM 服务的两种 API 调用方式 (无需真实服务)."""
from __future__ import annotations

import json


def show_native_api_call() -> None:
    """Ollama 原生 API: POST /api/chat."""
    print("--- Ollama 原生 API (POST /api/chat) ---")
    print("# 启动 Ollama 后: ollama run llama3.2:3b")
    print()
    print("import httpx")
    print("resp = httpx.post(")
    print('    "http://localhost:11434/api/chat",')
    print("    json={")
    print('        "model": "llama3.2:3b",')
    print('        "messages": [{"role": "user", "content": "Hello!"}],')
    print('        "stream": False,')
    print("    },")
    print("    timeout=60,")
    print(")")
    print("data = resp.json()")
    print('print(data["message"]["content"])  # 模型回复')


def show_openai_compat_call() -> None:
    """OpenAI 兼容 API: POST /v1/chat/completions - 可以用 openai SDK."""
    print("--- OpenAI 兼容 API (POST /v1/chat/completions) ---")
    print("用 openai SDK 调用本地 Ollama (改 base_url):")
    print()
    print("from openai import OpenAI")
    print("client = OpenAI(")
    print('    base_url="http://localhost:11434/v1",  # 关键: 指向本地 Ollama')
    print('    api_key="ollama",  # 任意占位')
    print(")")
    print("resp = client.chat.completions.create(")
    print('    model="llama3.2:3b",')
    print('    messages=[{"role": "user", "content": "Hello!"}],')
    print(")")
    print('print(resp.choices[0].message.content)')


def show_curl_examples() -> None:
    """curl 等价命令, 方便调试."""
    print("\n--- curl 等价命令 ---")
    print("# 原生 API:")
    print('curl -X POST http://localhost:11434/api/chat \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hi"}],"stream":false}\'')

    print("\n# OpenAI 兼容:")
    print('curl -X POST http://localhost:11434/v1/chat/completions \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hi"}]}\'')

    print("\n# 流式响应 (stream=true):")
    print('curl -N http://localhost:11434/api/chat -d \'{"model":"llama3.2:3b","stream":true,...}\'')


def show_pull_command() -> None:
    """常用模型管理命令."""
    print("\n--- 模型管理 ---")
    print("  ollama pull llama3.2:3b          # 拉取模型")
    print("  ollama list                      # 已下载模型")
    print("  ollama ps                        # 正在运行的模型")
    print("  ollama rm llama3.2:3b            # 删除模型")
    print("  ollama show llama3.2:3b          # 查看模型元数据")
    print("  ollama cp llama3.2:3b my-model   # 复制/重命名")


def main() -> None:
    show_native_api_call()
    print()
    show_openai_compat_call()
    show_curl_examples()
    show_pull_command()


if __name__ == "__main__":
    main()
