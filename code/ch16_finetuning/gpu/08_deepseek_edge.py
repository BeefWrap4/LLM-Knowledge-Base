# ---
# chapter: 16
# topic: DeepSeek Edge (真实 DeepSeek API via OpenAI 协议)
# section: 16.6.4
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: openai SDK
# run: export DEEPSEEK_API_KEY=sk-xxx; python 08_deepseek_edge.py
# expected_runtime: 5-30s (取决于网络与 V4 思考时间)
# expected_output: V4 Flash 非思考 + V4 Pro 思考两个真实响应
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.6.4
#
# Interview hooks:
#   1. DeepSeek 的 OpenAI 格式 API 与 OpenAI 原生 API 有哪些参数差异？
#   2. V4 Flash vs V4 Pro 如何选？同一模型如何切换 thinking？
#   3. 如何用线上评测衡量思考模式的质量、延迟与 token 成本？
"""DeepSeek Edge API 真实调用 (OpenAI 协议).

DeepSeek 提供 OpenAI 格式的 Chat Completions 接口，但参数集合并非与
OpenAI 原生 API 完全相同:
  base_url = https://api.deepseek.com
  api_key  = DEEPSEEK_API_KEY env
  model    = deepseek-v4-flash / deepseek-v4-pro

两个 V4 模型均支持思考与非思考模式；OpenAI SDK 需通过
``reasoning_effort`` 和 ``extra_body.thinking`` 显式控制思考。
"""

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import skip_if_mock


def get_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY":
        raise_with_help(
            "DEEPSEEK_API_KEY 未设置或为占位符",
            "运行 `export DEEPSEEK_API_KEY=sk-xxx` (Linux/Mac) "
            "或 `$env:DEEPSEEK_API_KEY='sk-xxx'` (PowerShell).",
        )
    return key


def main():
    if skip_if_mock("DeepSeek API 密钥、网络和可用额度"):
        return
    if os.environ.get("LLM_MOCK") != "0" or os.environ.get("DEEPSEEK_EDGE_RUN") != "1":
        print(
            "[SKIP] Real DeepSeek call requires both LLM_MOCK=0 and DEEPSEEK_EDGE_RUN=1 "
            "after reviewing model availability, quota, and cost."
        )
        return

    api_key = get_deepseek_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=== DeepSeek Edge API 真实调用 ===\n")

    # 1) V4 Flash — 显式关闭思考，保留可调采样参数
    print("[1/2] deepseek-v4-flash (thinking=disabled):")
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "讲个冷笑话, 一句话"}],
        max_tokens=64,
        temperature=0.7,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = resp.choices[0].message.content
    print(f"  回答: {content}")
    if resp.usage:
        print(
            f"  usage: total_tokens={resp.usage.total_tokens} "
            f"(prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens})"
        )

    # 2) V4 Pro — 显式开启思考；思考模式不使用 temperature/top_p 等采样参数
    print("\n[2/2] deepseek-v4-pro (thinking=enabled, reasoning_effort=high):")
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "9.11 和 9.9 哪个大? 请推理."}],
        max_tokens=512,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = resp.choices[0].message.content
    # 思考模式通过 reasoning_content 返回思考内容
    reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
    print(f"  回答: {content[:200]}")
    if reasoning:
        print(f"  推理过程: {reasoning[:200]}...")
    if resp.usage:
        print(f"  usage: total_tokens={resp.usage.total_tokens}")
    print("OK")


if __name__ == "__main__":
    main()
