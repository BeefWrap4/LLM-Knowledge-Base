#!/usr/bin/env python3
# ---
# code/scripts/llm_doctor.py
# 诊断所有已配置 API Key 的厂商 — 最小 token ping 测试
# Usage: python code/scripts/llm_doctor.py [--provider deepseek] [--all]
# Exit code: 0 全部通过 / 1 至少 1 家失败 / 2 无任何可用厂商
# ---
"""
LLM API Key 健康检查 — 对每个已配置 Key 的厂商发 1 个最小请求, 报告矩阵.

示例:
  $ python scripts/llm_doctor.py
  [✓] deepseek     chat "hi"  →  "Hi! How can I help?"   0.3s  (3 tok)
  [✓] kimi         chat "hi"  →  "你好! 有什么可以帮您?"  0.5s  (8 tok)
  [ ] siliconflow  (no API key)
  [ ] openai       (no API key)
  [✓] mock         (offline, no key needed)

  1/2 passed. Set LLM_PROVIDER=deepseek to use as default.
"""
import argparse
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from shared.provider_registry import (
    PROVIDERS, list_providers, get_provider, get_default_provider,
)
from shared.llm_client import UnifiedClient


def test_provider(name: str) -> tuple[bool, str, float, str]:
    """Returns (passed, output, elapsed_sec, error_msg)."""
    client = UnifiedClient(provider=name)
    if client.is_mock:
        return True, "(mock — no API call)", 0.0, ""

    t0 = time.perf_counter()
    try:
        resp = client.chat(
            prompt="Reply with exactly: 'OK'",
            max_tokens=10,
            temperature=0,
        )
        elapsed = time.perf_counter() - t0
        passed = "ok" in resp.content.lower() or len(resp.content) > 0
        return passed, resp.content.strip()[:60], elapsed, ""
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return False, "", elapsed, f"{type(e).__name__}: {str(e)[:80]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="只测指定厂商 (e.g. deepseek)")
    ap.add_argument("--all", action="store_true", help="测所有 6 个 (包括无 Key)")
    args = ap.parse_args()

    if args.provider:
        targets = [get_provider(args.provider)]
    elif args.all:
        targets = list_providers()
    else:
        # 默认: 只测有 Key 的
        targets = [p for p in PROVIDERS.values() if p.has_key()]
        if not targets:
            print("⚠️  无任何已配置 API Key. 复制 .env.example 为 .env 并填入.")
            print("    至少需要 1 个: DEEPSEEK_API_KEY / KIMI_API_KEY / SILICONFLOW_API_KEY")
            return 2

    print("=" * 60)
    print("  LLM DOCTOR — API Key 健康检查")
    print("=" * 60)
    print(f"\n  Targets: {[p.name for p in targets]}\n")

    passed = 0
    failed = 0
    results = []
    for p in targets:
        if p.name == "mock":
            # mock 永远通过
            mark = "✓"
            output = "(offline stub)"
            elapsed = 0.0
            err = ""
            results.append((p.name, True, output, elapsed, err))
            passed += 1
            continue

        ok, output, elapsed, err = test_provider(p.name)
        mark = "✓" if ok else "✗"
        results.append((p.name, ok, output, elapsed, err))
        if ok:
            passed += 1
        else:
            failed += 1
        # 实时打印
        display = output if ok else f"[{err}]"
        print(f"  [{mark}] {p.name:12s} ({p.region})  {display[:50]:50s}  {elapsed:.2f}s")

    print(f"\n{'=' * 60}")
    default = get_default_provider()
    print(f"  Default provider: {default.name} ({default.display_name})")
    print(f"  Result: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n  💡 提示:")
        print("    - 检查 API Key 是否过期: 重新登录厂商后台")
        print("    - 检查 base_url 是否可访问: 国内厂商通常无需代理")
        print("    - 切换厂商: LLM_PROVIDER=kimi python your_script.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
