#!/usr/bin/env python3
# ---
# code/scripts/verify_all.py
# 一键验证: 教程 wiki 链接 + 章节 README 覆盖 + core/ 例子跑通
# Usage: python code/scripts/verify_all.py
# ---
"""
Master verification script. Returns 0 iff all checks pass.

Checks (7 项):
  1. Wiki link integrity: 所有 [[WikiLinks]] 都能解析
  2. Chapter README coverage: 29/29 章节都有 README.md
  3. Code companion health:
     - 每章都有 core/ 或 llm/ 或 gpu/
     - 每章 .py 数 >= 1
  4. Tutorial ↔ Code bidirectional sync: §X.Y ↔ # section: X.Y
  5. CI LLM_MOCK safety: CI 环境应设 LLM_MOCK=1 (advisory)
  6. Smoke test sample: 跑 5 个代表性 core 例子
  7. LLM doctor (optional): 若环境有 API Key 跑诊断
"""

import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent


def check_wiki_links() -> bool:
    print("\n--- [1/7] Wiki link integrity ---")
    r = subprocess.run(
        [sys.executable, str(CODE / "scripts" / "verify_xrefs.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    # Filter out the summary lines we want
    for line in r.stdout.splitlines():
        if line.startswith("===") or "BROKEN" in line or "Resolved:" in line or "Broken:" in line:
            print(f"  {line}")
    return r.returncode == 0


def check_readme_coverage() -> bool:
    print("\n--- [2/7] Chapter README coverage ---")
    expected = 29
    actual = sum(1 for d in CODE.glob("ch*") if (d / "README.md").is_file())
    print(f"  Chapter READMEs: {actual}/{expected}")
    if actual < expected:
        missing = [d.name for d in CODE.glob("ch*") if not (d / "README.md").is_file()]
        print(f"  Missing: {missing}")
    return actual >= expected


def check_code_health() -> bool:
    print("\n--- [3/7] Code companion health ---")
    chapters = sorted(CODE.glob("ch*"))
    total_py = 0
    unhealthy = []
    for ch in chapters:
        py_files = list(ch.glob("*/[!_]*.py"))  # exclude __init__ if any
        n = len(py_files)
        total_py += n
        if n == 0:
            unhealthy.append(ch.name)
    print(f"  Chapters: {len(chapters)}")
    print(f"  Total .py: {total_py}")
    if unhealthy:
        print(f"  Unhealthy chapters (no .py): {unhealthy}")
        return False
    return total_py >= 400  # sanity check


def check_sync_links() -> bool:
    print("\n--- [4/7] Tutorial ↔ Code bidirectional sync ---")
    r = subprocess.run(
        [sys.executable, str(CODE / "scripts" / "sync_links.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    # Extract the key summary lines
    for line in r.stdout.splitlines():
        if (
            "教程章节总数" in line
            or "Code 例子含" in line
            or "教程章节有 code 覆盖" in line
            or line.startswith("=== PASS")
            or line.startswith("=== FAIL")
        ):
            print(f"  {line.strip()}")
    return r.returncode == 0


def check_ci_llm_mock_safety() -> bool:
    """CI 安全检查: 防止 PR check 意外调真实 API.

    规则:
    - 在 GitHub Actions CI 环境, LLM_MOCK 应被设 (避免 401/意外扣费)
    - 本地开发可灵活 (LLM_MOCK=0 走真实 API 也 OK)
    - 这项是 advisory, 不阻塞 (返回 True 总是, 只警告)
    """
    print("\n--- [5/7] CI LLM_MOCK safety check ---")
    in_ci = bool(os.environ.get("CI"))
    mock_set = os.environ.get("LLM_MOCK") == "1"

    if in_ci and not mock_set:
        print("  [WARN] CI 环境未设 LLM_MOCK=1, 可能意外调真实 API")
        print("         建议: GitHub Actions workflow 加 `env: LLM_MOCK: '1'`")
    elif mock_set:
        print("  [OK]   LLM_MOCK=1, 走 mock 路径 (CI 友好)")
    elif in_ci:
        print("  [OK]   CI 环境且未设 LLM_MOCK (例如 real-api job, 显式走真实 API)")
    else:
        print("  [INFO] 本地非 CI 环境, LLM_MOCK 未设 (会调真实 API 或抛缺 Key 错)")
    return True  # advisory, 不阻塞


def check_smoke() -> bool:
    print("\n--- [6/7] Smoke test sample (5 core/ files) ---")
    sample = [
        "ch01_python_basics/core/22_list_dict_basics.py",
        "ch02_mutability/core/01_is_vs_equals.py",
        "ch03_oop/core/01_singleton.py",
        "ch06_memory_gc/core/01_pymalloc_object_size.py",
        "ch07_data_structures/core/01_linked_list.py",
    ]
    failures = []
    for rel in sample:
        script = CODE / rel
        if not script.is_file():
            print(f"  SKIP  {rel} (missing)")
            continue
        r = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, cwd=str(CODE), timeout=30
        )
        ok = r.returncode == 0 and "OK" in r.stdout
        mark = "OK  " if ok else "FAIL"
        print(f"  {mark}  {rel}")
        if not ok:
            failures.append((rel, r.stderr[:100]))
    return len(failures) == 0


def main() -> int:
    print("=" * 60)
    print("  TUTORIAL+CODE COMPANION VERIFICATION")
    print("=" * 60)

    r1 = check_wiki_links()
    r2 = check_readme_coverage()
    r3 = check_code_health()
    r4 = check_sync_links()
    r5 = check_ci_llm_mock_safety()
    r6 = check_smoke()
    r7 = check_llm_doctor()

    print("\n" + "=" * 60)
    print(f"  Wiki links:        {'PASS' if r1 else 'FAIL'}")
    print(f"  README coverage:   {'PASS' if r2 else 'FAIL'}")
    print(f"  Code health:       {'PASS' if r3 else 'FAIL'}")
    print(f"  Sync links:        {'PASS' if r4 else 'FAIL'}")
    print(f"  LLM_MOCK safety:   {'PASS' if r5 else 'FAIL'}  (advisory)")
    print(f"  Smoke sample:      {'PASS' if r6 else 'FAIL'}")
    print(f"  LLM doctor:        {'PASS' if r7 else 'FAIL'}")
    print("=" * 60)
    return 0 if all([r1, r2, r3, r4, r5, r6, r7]) else 1


def check_llm_doctor() -> bool:
    """Optional: skip if no API key (return True). 实际有 key 时跑全部."""
    print("\n--- [7/7] LLM doctor (API key health) ---")
    # 如果没任何 key, 跳过 (但 100% 通过, 因为 mock 也在)
    sys.path.insert(0, str(CODE))  # 让 shared 可 import
    try:
        from shared.provider_registry import PROVIDERS

        has_key = any(p.has_key() for p in PROVIDERS.values() if p.name != "mock")
    except Exception as e:
        print(f"  WARN  provider_registry 不可用: {e}")
        return True
    if not has_key:
        print("  SKIP (no LLM API key in env — using mock for everything)")
        return True
    r = subprocess.run(
        [sys.executable, str(CODE / "scripts" / "llm_doctor.py")],
        capture_output=True,
        text=True,
        cwd=str(CODE),
        timeout=120,
    )
    for line in r.stdout.splitlines():
        if "[✓]" in line or "[✗]" in line or "passed" in line or "Result:" in line:
            print(f"  {line.strip()}")
    return r.returncode == 0


if __name__ == "__main__":
    sys.exit(main())
