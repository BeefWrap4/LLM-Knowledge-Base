#!/usr/bin/env python3
# ---
# code/scripts/test_real_api_smoke.py
# 单 provider 真实 API 最小冒烟；不会把模板、mock、skip 或空输出计为通过
# Usage: LLM_MOCK=0 python scripts/test_real_api_smoke.py --provider deepseek --confirm-real
# Exit code: 0 严格通过，1 请求/断言失败，2 门禁或配置不完整
# ---
"""对一个明确 provider 执行一次可能计费的严格真实 API 探针。

该脚本不再批量运行 ``llm/`` 教学模板，因为进程退出码 0 无法证明模板真的访问了 API。
真实请求必须同时满足精确 ``LLM_MOCK=0``、``--provider`` 和 ``--confirm-real``。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from shared.llm_client import UnifiedClient  # noqa: E402
from shared.provider_registry import get_provider  # noqa: E402


def _is_usable_key(value: str) -> bool:
    normalized = value.strip()
    return bool(normalized) and normalized != "YOUR_API_KEY" and not normalized.startswith(
        ("sk-placeholder", "your-", "test-")
    )


def _normalize_probe_output(content: str) -> str:
    return content.strip().strip("'\"` \t\r\n.!。").casefold()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="严格真实 LLM API 冒烟")
    parser.add_argument("--provider", required=True, help="明确指定一个非 mock provider")
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="确认允许一次可能计费的真实请求（仍必须 LLM_MOCK=0）",
    )
    return parser


def validate_gate(provider_name: str, confirm_real: bool) -> tuple[bool, str]:
    if os.environ.get("LLM_MOCK") != "0":
        return False, "LLM_MOCK 必须精确为 0"
    if not confirm_real:
        return False, "缺少 --confirm-real"
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        return False, "真实计费冒烟不允许在 pull_request 事件运行"

    try:
        provider = get_provider(provider_name)
    except ValueError as exc:
        return False, str(exc)
    if provider.name == "mock":
        return False, "mock 不能作为真实验收 provider"

    env_provider = os.environ.get("LLM_PROVIDER", "").strip()
    if env_provider and env_provider.casefold() != provider.name.casefold():
        return False, f"LLM_PROVIDER={env_provider!r} 与 --provider {provider.name!r} 不一致"

    # has_key() 在精确真实模式下负责按仓库规则加载 .env。
    if not provider.has_key():
        return False, f"缺少 {provider.env_key}"
    if not _is_usable_key(os.environ.get(provider.env_key, "")):
        return False, f"{provider.env_key} 为空或是占位符"
    return True, ""


def run_probe(provider_name: str) -> tuple[bool, str]:
    provider = get_provider(provider_name)
    started = time.perf_counter()
    try:
        client = UnifiedClient(provider=provider.name, timeout=30.0)
        if client.is_mock:
            return False, "创建了 mock client"
        response = client.chat(
            prompt="Reply with exactly: OK",
            max_tokens=16,
            temperature=0,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:180]}"

    elapsed = time.perf_counter() - started
    if response.mock:
        return False, "响应被标记为 mock"
    if response.raw is None:
        return False, "响应 raw 为空，缺少真实 SDK 返回证据"
    if _normalize_probe_output(response.content) != "ok":
        return False, f"业务断言失败：期望 OK，实际 {response.content[:80]!r}"

    usage = response.usage if isinstance(response.usage, dict) else {}
    print(
        f"[PASS] provider={response.provider} model={response.model} "
        f"elapsed={elapsed:.2f}s usage={usage}"
    )
    return True, ""


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gate_ok, reason = validate_gate(args.provider, args.confirm_real)
    if not gate_ok:
        print(f"[ERROR] 拒绝真实 API 冒烟：{reason}")
        print(
            "        示例：LLM_MOCK=0 LLM_PROVIDER=deepseek "
            "python scripts/test_real_api_smoke.py --provider deepseek --confirm-real"
        )
        return 2

    ok, error = run_probe(args.provider)
    if not ok:
        print(f"[FAIL] {args.provider}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
