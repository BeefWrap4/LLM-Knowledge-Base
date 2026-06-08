#!/usr/bin/env python3
# ---
# code/scripts/run_all_examples.py
# 跑完所有 code/chNN_*/core/*.py 例子, 报告通过/失败
# Usage: python code/scripts/run_all_examples.py [--tier core|llm|gpu] [--chapter chNN]
# Exit code: 0 if all OK, 1 if any failure
# ---
"""
实际执行所有 .py 例子, 捕获输出与返回码.

策略:
  - core/ 全跑 (任意机器可跑, 30s 安装)
  - llm/ 默认 mock 模式 (无 API key)
  - gpu/ 默认 skip (无 CUDA)
  - 单个文件超时 30s 防止 hang
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
PY = sys.executable


def discover(tier: str) -> list[Path]:
    """Find all .py files for the given tier under code/ch*/*/."""
    return sorted(CODE.glob(f"ch*/{tier}/*.py"))


def run_one(script: Path, timeout: int = 30) -> tuple[str, bool, str, float]:
    """Returns (rel_path, passed, output, elapsed)."""
    rel = str(script.relative_to(CODE))
    t0 = time.perf_counter()
    try:
        # GPU 例子默认需要 --mock 标志 (无真实模型)
        # 跨平台: 同时检查 /gpu/ 和 \gpu\
        cmd = [PY, str(script)]
        if "/gpu/" in rel or "\\gpu\\" in rel:
            cmd.append("--mock")
        # 显式传 env (确保 LLM_MOCK=1 等 CI 变量传到子进程)
        # 跨平台编码强制 UTF-8 (避免 Windows cp936 vs Linux utf-8 差异)
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(CODE),
            env=child_env,
        )
        elapsed = time.perf_counter() - t0
        passed = result.returncode == 0
        # 失败时多留 stderr (CI 调试关键)
        if passed:
            out = (result.stdout or "")[-100:] + (result.stderr or "")[-100:]
        else:
            out = (result.stderr or "[no stderr]")[-800:] + "\n--- stdout ---\n" + (result.stdout or "")[-200:]
        return rel, passed, out, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return rel, False, f"[TIMEOUT after {timeout}s]", elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return rel, False, f"[ERROR: {e}]", elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="core", choices=["core", "llm", "gpu"])
    ap.add_argument("--chapter", default=None, help="e.g. ch12")
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="per-file timeout in seconds (some HF model loads take 60s+)",
    )
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    files = discover(args.tier)
    if args.chapter:
        files = [f for f in files if f.parent.parent.name == args.chapter]
    if not files:
        print(f"No {args.tier} files found", file=sys.stderr)
        return 1

    print(f"=== Running {len(files)} {args.tier}/ examples ===")
    print(f"Timeout: {args.timeout}s, Parallel: {args.parallel}\n")

    results: list[tuple[str, bool, str, float]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_one, f, args.timeout): f for f in files}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            rel, ok, _, _ = fut.result()
            mark = "OK " if ok else "FAIL"
            print(f"  [{done:3d}/{len(files)}] {mark}  {rel}", flush=True)
    total_elapsed = time.perf_counter() - t0

    # Summary
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = len(results) - passed
    print("\n=== Summary ===")
    print(f"Total:   {len(results)}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Time:    {total_elapsed:.1f}s")

    if failed:
        print("\n--- Failures ---")
        for rel, ok, out, _ in results:
            if not ok:
                print(f"\n  {rel}")
                print(f"  | {out.strip()[:200]}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
