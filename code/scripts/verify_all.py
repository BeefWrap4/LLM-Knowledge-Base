#!/usr/bin/env python3
# ---
# code/scripts/verify_all.py
# 一键验证: 教程 wiki 链接 + 章节 README 覆盖 + core/ 例子跑通
# Usage: python code/scripts/verify_all.py
# ---
"""
Master verification script. Returns 0 iff all checks pass.

Checks:
  1. Wiki link integrity: 所有 [[WikiLinks]] 都能解析
  2. Chapter README coverage: 29/29 章节都有 README.md
  3. Code companion health:
     - 每章都有 core/ 或 llm/ 或 gpu/
     - 每章 .py 数 >= 1
  4. Smoke test sample: 跑 5 个代表性 core 例子
"""
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent


def check_wiki_links() -> bool:
    print("\n--- [1/5] Wiki link integrity ---")
    r = subprocess.run([sys.executable, str(CODE / "scripts" / "verify_xrefs.py")],
                       capture_output=True, text=True, cwd=str(REPO))
    # Filter out the summary lines we want
    for line in r.stdout.splitlines():
        if line.startswith("===") or "BROKEN" in line or "Resolved:" in line or "Broken:" in line:
            print(f"  {line}")
    return r.returncode == 0


def check_readme_coverage() -> bool:
    print("\n--- [2/5] Chapter README coverage ---")
    expected = 29
    actual = sum(1 for d in CODE.glob("ch*") if (d / "README.md").is_file())
    print(f"  Chapter READMEs: {actual}/{expected}")
    if actual < expected:
        missing = [d.name for d in CODE.glob("ch*") if not (d / "README.md").is_file()]
        print(f"  Missing: {missing}")
    return actual >= expected


def check_code_health() -> bool:
    print("\n--- [3/5] Code companion health ---")
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
    print("\n--- [4/5] Tutorial ↔ Code bidirectional sync ---")
    r = subprocess.run([sys.executable, str(CODE / "scripts" / "sync_links.py")],
                       capture_output=True, text=True, cwd=str(REPO))
    # Extract the key summary lines
    for line in r.stdout.splitlines():
        if "教程章节总数" in line or "Code 例子含" in line or "教程章节有 code 覆盖" in line \
           or line.startswith("=== PASS") or line.startswith("=== FAIL"):
            print(f"  {line.strip()}")
    return r.returncode == 0


def check_smoke() -> bool:
    print("\n--- [5/5] Smoke test sample (5 core/ files) ---")
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
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           cwd=str(CODE), timeout=30)
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
    r5 = check_smoke()

    print("\n" + "=" * 60)
    print(f"  Wiki links:        {'PASS' if r1 else 'FAIL'}")
    print(f"  README coverage:   {'PASS' if r2 else 'FAIL'}")
    print(f"  Code health:       {'PASS' if r3 else 'FAIL'}")
    print(f"  Sync links:        {'PASS' if r4 else 'FAIL'}")
    print(f"  Smoke sample:      {'PASS' if r5 else 'FAIL'}")
    print("=" * 60)
    return 0 if all([r1, r2, r3, r4, r5]) else 1


if __name__ == "__main__":
    sys.exit(main())
