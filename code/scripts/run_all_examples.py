#!/usr/bin/env python3
# ---
# code/scripts/run_all_examples.py
# 跑完所有 code/chNN_*/core/*.py 例子, 报告通过/失败
# Usage: python code/scripts/run_all_examples.py [--tier core|llm|gpu] [--chapter chNN] [--real-gpu]
# Exit code: 0 if all OK, 1 if any failure
# ---
"""
实际执行所有 .py 例子, 捕获输出与返回码.

策略:
  - core/ 全跑 (任意机器可跑, 30s 安装)
  - llm/ 默认 mock 模式 (无 API key)
  - gpu/ 默认传 --mock，条件项计为 SKIP；--real-gpu 才进入真实路径
  - 单个文件超时 30s 防止 hang
"""

import argparse
import os
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


def run_one(
    script: Path,
    timeout: int = 30,
    tier: str | None = None,
    real_gpu: bool = False,
) -> tuple[str, bool, str, float]:
    """Returns (rel_path, passed, output, elapsed)."""
    rel = str(script.relative_to(CODE))
    t0 = time.perf_counter()
    try:
        # GPU tier 跨 NVIDIA/Apple/本地服务；默认用 --mock 阻断外部副作用。
        # 跨平台: 同时检查 /gpu/ 和 \gpu\
        cmd = [PY, str(script)]
        if tier == "gpu" and not real_gpu:
            cmd.append("--mock")
        # 显式传 env (确保 LLM_MOCK=1 等 CI 变量传到子进程)
        # 跨平台编码强制 UTF-8 (避免 Windows cp936 vs Linux utf-8 差异)
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # llm/ 批量 runner 的契约始终是离线、确定性运行。即使父进程设了
        # LLM_MOCK=0，也必须覆盖为 1；真实 API 只能走带确认门禁的单独 runner。
        if tier == "llm":
            child_env["LLM_MOCK"] = "1"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(CODE),
            env=child_env,
        )
        elapsed = time.perf_counter() - t0
        combined_output = (result.stdout or "") + (result.stderr or "")
        passed = result.returncode == 0 and "OK" in combined_output
        # 失败时多留 stderr (CI 调试关键)
        if passed:
            out = (result.stdout or "")[-100:] + (result.stderr or "")[-100:]
            if "[SKIP]" in combined_output:
                # 保留结构化状态；长前置条件说明不能把标记截出摘要窗口。
                out = "[SKIP]\n" + out
        else:
            reason = (
                "[MISSING OK MARKER]\n"
                if result.returncode == 0 and "OK" not in combined_output
                else ""
            )
            out = (
                reason
                + (result.stderr or "[no stderr]")[-800:]
                + "\n--- stdout ---\n"
                + (result.stdout or "")[-200:]
            )
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
    ap.add_argument("--chapter", default=None, help="chapter prefix or directory, e.g. ch12")
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="per-file timeout in seconds (some HF model loads take 60s+)",
    )
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument(
        "--real-gpu",
        action="store_true",
        help="do not append --mock to gpu examples; use only on a compatible, isolated GPU host",
    )
    args = ap.parse_args()
    if args.parallel < 1:
        ap.error("--parallel must be at least 1")
    if args.real_gpu and args.tier != "gpu":
        ap.error("--real-gpu is only valid with --tier gpu")
    if args.real_gpu and not args.chapter:
        ap.error("--real-gpu requires an explicit --chapter scope")
    if args.real_gpu and args.parallel != 1:
        ap.error("--real-gpu requires --parallel 1 to avoid concurrent services or resource creation")

    files = discover(args.tier)
    if args.chapter:
        chapter = args.chapter.lower()
        files = [
            f
            for f in files
            if f.parent.parent.name.lower() == chapter
            or f.parent.parent.name.lower().startswith(f"{chapter}_")
        ]
    if not files:
        print(f"No {args.tier} files found", file=sys.stderr)
        return 1

    print(f"=== Running {len(files)} {args.tier}/ examples ===")
    print(f"Timeout: {args.timeout}s, Parallel: {args.parallel}\n")

    results: list[tuple[str, bool, str, float]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(run_one, f, args.timeout, args.tier, args.real_gpu): f for f in files
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            rel, ok, _, _ = fut.result()
            skipped = ok and "[SKIP]" in fut.result()[2]
            mark = "SKIP" if skipped else ("OK  " if ok else "FAIL")
            print(f"  [{done:3d}/{len(files)}] {mark}  {rel}", flush=True)
    total_elapsed = time.perf_counter() - t0

    # Summary
    skipped = sum(1 for _, ok, out, _ in results if ok and "[SKIP]" in out)
    passed = sum(1 for _, ok, out, _ in results if ok and "[SKIP]" not in out)
    failed = len(results) - passed - skipped
    print("\n=== Summary ===")
    print(f"Total:   {len(results)}")
    print(f"Passed:  {passed}")
    print(f"Skipped: {skipped}")
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
