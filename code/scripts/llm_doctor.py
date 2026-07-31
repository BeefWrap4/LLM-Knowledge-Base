#!/usr/bin/env python3
# ---
# code/scripts/llm_doctor.py
# LLM provider 配置检查与最小真实 API 探针
# Usage: python scripts/llm_doctor.py [--check|--provider NAME|--all] --confirm-real
# Exit code: 0 通过/仅查看配置，1 至少一个真实探针失败，2 门禁或配置不完整
# ---
"""安全诊断 LLM provider。

无参数运行只显示说明，不读取密钥、不发请求。真实探针必须同时满足：

1. 精确设置 ``LLM_MOCK=0``；
2. 显式传入 ``--confirm-real``；
3. 选择 ``--check``、``--provider NAME`` 或 ``--all``。

``--setup`` 只写入本地 ``code/.env``。只有额外满足上述真实门禁时，才会在保存后执行探针。
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from shared.llm_client import UnifiedClient  # noqa: E402
from shared.provider_registry import Provider, get_provider, list_providers  # noqa: E402


def _normalize_probe_output(content: str) -> str:
    """Normalize harmless quoting/punctuation around the expected ``OK``."""
    return content.strip().strip("'\"` \t\r\n.!。").casefold()


def _is_usable_key(value: str) -> bool:
    normalized = value.strip()
    return bool(normalized) and normalized != "YOUR_API_KEY" and not normalized.startswith(
        ("sk-placeholder", "your-", "test-")
    )


def _real_gate_is_open(confirm_real: bool) -> bool:
    return confirm_real and os.environ.get("LLM_MOCK") == "0"


def _print_real_gate_error() -> None:
    print("[ERROR] 拒绝真实 API 请求。")
    print("        必须同时设置 LLM_MOCK=0 并传入 --confirm-real。")
    print("        该探针可能产生费用；请先确认 provider、模型、配额和预算。")


def test_provider(name: str, api_key: str | None = None) -> tuple[bool, str, float, str]:
    """Run one paid probe and return ``(passed, output, elapsed, error)``."""
    started = time.perf_counter()
    try:
        client = UnifiedClient(provider=name, api_key=api_key, timeout=30.0)
        if client.is_mock:
            return False, "", 0.0, "拒绝把 mock client 计为真实 API 通过"
        response = client.chat(
            prompt="Reply with exactly: OK",
            max_tokens=16,
            temperature=0,
        )
        elapsed = time.perf_counter() - started
        if response.mock or response.raw is None:
            return False, "", elapsed, "响应缺少真实调用证据（mock 或 raw 为空）"
        output = response.content.strip()
        if _normalize_probe_output(output) != "ok":
            return False, output[:80], elapsed, "响应未严格满足探针断言：期望 OK"
        return True, output[:80], elapsed, ""
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return False, "", elapsed, f"{type(exc).__name__}: {str(exc)[:160]}"


def _configured_targets() -> list[Provider]:
    """Return configured real providers; only called after the real-mode gate."""
    return [provider for provider in list_providers() if provider.name != "mock" and provider.has_key()]


def _run_targets(targets: list[Provider]) -> int:
    if not targets:
        print("[ERROR] 未找到可测试的 provider 或有效 Key。")
        return 2

    print("=" * 68)
    print("LLM DOCTOR — 真实 API 最小探针（可能计费）")
    print("=" * 68)

    failed = 0
    for provider in targets:
        if not provider.has_key() or not _is_usable_key(os.environ.get(provider.env_key, "")):
            print(f"[FAIL] {provider.name:12s} 缺少 {provider.env_key}")
            failed += 1
            continue

        ok, output, elapsed, error = test_provider(
            provider.name,
            api_key=os.environ[provider.env_key],
        )
        if ok:
            print(
                f"[PASS] {provider.name:12s} model={provider.default_chat} "
                f"response={output!r} elapsed={elapsed:.2f}s"
            )
        else:
            print(f"[FAIL] {provider.name:12s} elapsed={elapsed:.2f}s error={error}")
            failed += 1

    print("-" * 68)
    print(f"结果：{len(targets) - failed} passed, {failed} failed")
    return 1 if failed else 0


def _write_env_key(provider: Provider, api_key: str) -> Path:
    env_path = CODE / ".env"
    line = f"{provider.env_key}={api_key}"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        pattern = rf"^{re.escape(provider.env_key)}=.*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, line, content, flags=re.MULTILINE)
        else:
            content = content.rstrip("\n") + "\n" + line + "\n"
    else:
        content = line + "\n"
    env_path.write_text(content, encoding="utf-8")
    return env_path


def setup_wizard() -> tuple[Provider, str] | None:
    """Prompt for one key and save it locally without making a request."""
    real_providers = [provider for provider in list_providers() if provider.name != "mock"]
    print("可配置的 provider：")
    for index, provider in enumerate(real_providers, start=1):
        print(f"  {index}. {provider.name} ({provider.env_key})")

    try:
        raw_choice = input("选择编号（q 取消）：").strip()
    except EOFError:
        print("[INFO] 非交互环境，取消 setup。")
        return None
    if raw_choice.casefold() == "q":
        return None
    try:
        provider = real_providers[int(raw_choice) - 1]
    except (ValueError, IndexError):
        print("[ERROR] 无效编号。")
        return None

    try:
        api_key = getpass.getpass(f"粘贴 {provider.env_key}（输入不回显）：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] 已取消，未写入 Key。")
        return None
    if not _is_usable_key(api_key):
        print("[ERROR] Key 为空或是占位符，未写入。")
        return None

    env_path = _write_env_key(provider, api_key)
    print(f"[OK] 已保存 {provider.env_key} 到 {env_path}；未显示或验证密钥。")
    return provider, api_key


def print_report() -> None:
    """Print a network-free provider overview."""
    print("LLM provider 配置检查（无网络请求）")
    print("默认未设置 LLM_MOCK 时保持离线；本命令不会把离线结果计为真实通过。")
    for provider in list_providers():
        if provider.name != "mock":
            print(
                f"  - {provider.name:12s} env={provider.env_key:24s} "
                f"default={provider.default_chat}"
            )
    print("真实探针：LLM_MOCK=0 python scripts/llm_doctor.py --provider NAME --confirm-real")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM provider 配置检查与真实 API 最小探针")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--provider", help="只测试一个明确的 provider")
    actions.add_argument("--all", action="store_true", help="测试全部已注册 provider；缺 Key 计失败")
    actions.add_argument("--check", action="store_true", help="测试所有已配置 Key 的 provider")
    actions.add_argument("--setup", action="store_true", help="交互写入一个 Key；默认不发请求")
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="确认允许一次或多次可能计费的真实请求（仍必须 LLM_MOCK=0）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.setup:
        configured = setup_wizard()
        if configured is None:
            return 2
        provider, api_key = configured
        if not args.confirm_real:
            print("[INFO] setup 已完成；未传 --confirm-real，因此没有调用 API。")
            return 0
        if not _real_gate_is_open(args.confirm_real):
            _print_real_gate_error()
            return 2
        ok, output, elapsed, error = test_provider(provider.name, api_key=api_key)
        if ok:
            print(f"[PASS] {provider.name} response={output!r} elapsed={elapsed:.2f}s")
            return 0
        print(f"[FAIL] {provider.name} elapsed={elapsed:.2f}s error={error}")
        return 1

    requested_probe = bool(args.provider or args.all or args.check)
    if not requested_probe:
        print_report()
        return 0
    if not _real_gate_is_open(args.confirm_real):
        _print_real_gate_error()
        return 2

    if args.provider:
        try:
            targets = [get_provider(args.provider)]
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return 2
        if targets[0].name == "mock":
            print("[ERROR] mock 不能作为真实 API 验收 provider。")
            return 2
    elif args.all:
        targets = [provider for provider in list_providers() if provider.name != "mock"]
    else:
        targets = _configured_targets()

    return _run_targets(targets)


if __name__ == "__main__":
    sys.exit(main())
