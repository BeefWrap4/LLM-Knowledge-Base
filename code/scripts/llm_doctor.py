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
import re
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from shared.llm_client import UnifiedClient
from shared.provider_registry import (
    PROVIDERS,
    get_default_provider,
    get_provider,
    list_providers,
)


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
    ap = argparse.ArgumentParser(description="LLM API Key 诊断 / 配置工具")
    ap.add_argument("--provider", help="只测指定厂商 (e.g. deepseek)")
    ap.add_argument("--all", action="store_true", help="测所有 6 个 (包括无 Key)")
    ap.add_argument("--setup", action="store_true", help="交互式引导配置 API Key")
    ap.add_argument("--check", action="store_true", help="测试已配置 Key 是否有效 (实际调一次最小 API)")
    args = ap.parse_args()

    # 短命令路由: --setup / --check 走独立流程
    if args.setup:
        setup_wizard()
        return 0
    if args.check:
        check_keys()
        return 0

    # 现有 report 行为 (--provider/--all/无参)
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
            print("    或运行 `python scripts/llm_doctor.py --setup` 交互式配置.")
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


# ────────────────────────────────────────────────────────────────
# W1-T7: --setup / --check 交互式命令
# ────────────────────────────────────────────────────────────────


def setup_wizard():
    """交互式引导: 帮用户配置 API Key."""
    print("=" * 60)
    print("LLM API Key 配置向导")
    print("=" * 60)
    print()
    print("教程默认使用 DeepSeek (国内访问快 + 注册送 ¥10 + OpenAI 协议).")
    print("注册地址: https://platform.deepseek.com")
    print()
    print("其他可选厂商:")
    for p in list_providers():
        if p.name in ("deepseek", "mock"):
            continue
        print(f"  - {p.display_name} ({p.name}): {p.free_tier}")
    print()

    try:
        choice = input("选择厂商 [1=DeepSeek, 2=其他, q=退出]: ").strip()
    except EOFError:
        print("\n[INFO] 非交互环境, 跳过 setup. 使用 `python scripts/llm_doctor.py --check` 验证.")
        return
    if choice == "q":
        return
    if choice not in ("1", "2"):
        print("无效选择.")
        return

    if choice == "1":
        provider = "deepseek"
        env_var = "DEEPSEEK_API_KEY"
        print("\n请访问 https://platform.deepseek.com 注册并获取 API Key.")
    else:
        names = [p.name for p in list_providers() if p.name not in ("mock", "deepseek")]
        for i, n in enumerate(names, 1):
            print(f"  {i}. {n}")
        try:
            idx = int(input("选编号: ")) - 1
        except (EOFError, ValueError):
            print("\n[INFO] 非交互环境, 退出.")
            return
        if idx < 0 or idx >= len(names):
            print("无效选择.")
            return
        provider = names[idx]
        env_var = get_provider(provider).env_key

    try:
        api_key = input(f"粘贴 {env_var} (输入时不会显示): ").strip()
    except EOFError:
        print("\n[INFO] 非交互环境, 退出.")
        return

    if not api_key:
        print("[WARN] 未输入 Key, 取消写入.")
        return

    # 写入 .env (向上查找 3 层以匹配 shared/env.py 的行为)
    env_path = CODE / ".env"
    line = f"{env_var}={api_key}\n"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if re.search(rf"^{re.escape(env_var)}=.*$", content, flags=re.M):
            content = re.sub(
                rf"^{re.escape(env_var)}=.*$",
                line.rstrip(),
                content,
                flags=re.M,
            )
        else:
            content += "\n" + line
        env_path.write_text(content, encoding="utf-8")
    else:
        env_path.write_text(line, encoding="utf-8")

    print(f"\n[OK] 已写入 {env_path}")
    print(f"     {env_var}={'*' * 8}{api_key[-4:]}")

    # 测试连通性
    print("\n测试调用...")
    result = test_provider_with_key(provider, api_key)
    if result["ok"]:
        print(f"[OK] {provider} 可用 (延迟 {result['latency_ms']}ms)")
    else:
        print(f"[FAIL] {provider} 失败: {result['error']}")


def test_provider_with_key(provider: str, api_key: str) -> dict:
    """测试厂商 API 是否可用 (传入显式 key, 用于 --check / --setup).

    与 module 顶部 test_provider(name) 区别: 这里接受显式 api_key,
    避免污染环境变量.

    失败检测: UnifiedClient 在 API 出错时不抛, 而是 fallback 到 mock.
    因此我们额外检查 resp.mock / resp.raw: 两者皆为 fallback 特征,
    代表 API 实际失败 (auth / 网络 / 超时), 即使没崩溃也应标 FAIL.
    """
    client = UnifiedClient(provider=provider, api_key=api_key)
    t0 = time.perf_counter()
    try:
        resp = client.chat(prompt="Reply with exactly: 'OK'", max_tokens=10, temperature=0)
        latency = (time.perf_counter() - t0) * 1000
        # resp.mock=True → UnifiedClient fallback 到了 mock (API 实际失败)
        if getattr(resp, "mock", False):
            return {
                "ok": False,
                "error": "API fallback to mock (auth/network error, see stderr)",
                "latency_ms": round(latency),
            }
        return {"ok": True, "latency_ms": round(latency)}
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)[:120]}",
            "latency_ms": round(latency),
        }


def check_keys():
    """列出已配置 Key 并测试连通性."""
    ap = [p for p in PROVIDERS.values() if p.has_key()]
    print("=" * 60)
    print("  LLM DOCTOR — 已配置 Key 健康检查")
    print("=" * 60)
    if not ap:
        print("\n  (无). 运行 `python scripts/llm_doctor.py --setup` 配置.")
        return
    print()
    for p in ap:
        print(f"  [test] {p.name:12s} ({p.display_name})")
        result = test_provider_with_key(p.name, os.environ.get(p.env_key, ""))
        if result["ok"]:
            print(f"     [OK]  延迟 {result['latency_ms']}ms")
        else:
            print(f"     [FAIL] {result['error']}")


def print_report():
    """Provider 状态简表 (--help 时 fallback 或独立调用)."""
    print("=" * 60)
    print("  LLM DOCTOR — Provider 状态")
    print("=" * 60)
    for p in list_providers():
        if p.name == "mock":
            continue
        status = "[OK]" if p.has_key() else "[  ]"
        print(f"  {status} {p.name:12s} ({p.region})  {p.display_name}")
    if not [p for p in PROVIDERS.values() if p.has_key()]:
        print("\n[INFO] 未配置 API Key. 运行 `python scripts/llm_doctor.py --setup`.")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
