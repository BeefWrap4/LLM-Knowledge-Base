# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: edge_llm.webgpu_vs_wasm
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: wasmtime CLI (winget install BytecodeAlliance.Wasmtime.Portable)
# run: python 08_webgpu_vs_wasm.py
# expected_runtime: ~30ms (1000 wasm add(1,2) assertions)
# expected_output: 真实 wasmtime CLI 跑分 + 对比决策表
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
# Interview hooks:
#   1. WebGPU 和 WebAssembly 各自最适合什么场景?
#   2. WebGPU 为什么有 4-8GB 显存限制 (来自 WebGL/Vulkan 抽象层)?
#   3. 量化等级对 WebGPU 性能的影响?
"""WebGPU vs WebAssembly 浏览器推理对比 + 真实 wasmtime CLI 跑分.

本文件做两件事:
  1. 真实跑 wasmtime CLI: 写一个简单 WAT 模块, 用 wast runner
     做 N 次 add(1,2) 断言, 测量 wasmtime 调用延迟
  2. 打印 WebGPU vs WASM 对比矩阵 + 决策流程 (教学保留)

WebGPU 部分需 GPU + 浏览器交互, 本地不跑 (仅文字对比).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 让脚本既能 `python file.py` 也能 `import` 找到 shared/
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help  # noqa: E402
from shared.gpu_guard import skip_if_mock  # noqa: E402

# ============================================================
# 1. WebGPU vs WASM 对比矩阵 (教学)
# ============================================================
COMPARISON_MATRIX = [
    ("性能", "⭐⭐⭐⭐⭐", "⭐⭐⭐", "WebGPU 利用 GPU 并行"),
    ("浏览器兼容", "Chromium/Safari", "全部浏览器", "WASM 通用性更强"),
    ("计算单元", "GPU (并行)", "CPU (串行)", "本质区别"),
    ("API 风格", "现代 (wgpu)", "低级 (LLVM IR)", "WebGPU 抽象更友好"),
    ("内存", "GPU 显存 4-8GB", "系统 RAM", "WebGPU 受限于安全沙箱"),
    ("适用模型大小", "1B-7B 量化", "<1B", "WASM 算力天花板"),
    ("功耗", "中-高 (GPU)", "低 (CPU)", "WASM 适合移动端续航"),
    ("首推框架", "WebLLM/MLC-LLM", "WasmEdge/llama.cpp", "生态成熟度"),
]


def print_comparison() -> None:
    print(f"{'维度':<14} {'WebGPU':<25} {'WebAssembly':<20} {'说明'}")
    print("-" * 90)
    for dim, wgpu, wasm, note in COMPARISON_MATRIX:
        print(f"{dim:<14} {wgpu:<25} {wasm:<20} {note}")


def decision_tree() -> None:
    """根据场景选择 WebGPU 还是 WASM."""
    print("\n--- 决策流程 ---")
    print("""
1. 模型大小?
   - <1B      → WebAssembly (WasmEdge / llama.cpp.wasm)
   - 1B-7B    → WebGPU (WebLLM / MLC-LLM)
   - >7B      → 云端推理

2. 浏览器?
   - 必须是 Firefox/老 Safari → WebAssembly
   - Chrome/Edge/Safari 17+   → WebGPU 优先

3. 设备?
   - 桌面 GPU         → WebGPU 全速
   - 笔记本集显       → WebGPU 仍优
   - 手机 (iOS Safari)→ WebGPU 也能用, 性能约桌面 30%

4. 隐私/离线?
   - 任何浏览器推理方案都满足本地处理, 选最契合模型的即可
""")


# ============================================================
# 2. 真实 wasmtime 跑分
# ============================================================
WASM_ADD_WAT = """\
(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add)
  (export "add" (func $add)))
"""


def find_wasmtime() -> str:
    """查找 wasmtime 可执行文件路径 (PATH → WASMTIME_EXE → 标准安装位置)."""
    # 1) 优先 PATH
    p = shutil.which("wasmtime")
    if p:
        return p
    # 2) 优先环境变量
    p = os.environ.get("WASMTIME_EXE")
    if p and Path(p).is_file():
        return p
    # 3) Windows 标准 winget portable 安装位置
    winget_dir = Path(os.environ.get("LOCALAPPDATA", "")) / ("Microsoft/WinGet/Packages")
    if winget_dir.is_dir():
        for d in winget_dir.glob("BytecodeAlliance.Wasmtime*"):
            for exe in d.rglob("wasmtime.exe"):
                return str(exe)
    # 4) 找不到 → 抛错
    raise_with_help(
        "wasmtime CLI 未装.",
        "运行 `winget install --id BytecodeAlliance.Wasmtime.Portable "
        "--accept-package-agreements --accept-source-agreements` (Windows) "
        "或 `curl https://wasmtime.dev/install.sh | bash` (Linux/Mac). "
        "或设置环境变量 WASMTIME_EXE 指向 wasmtime.exe 完整路径.",
    )


def run_wasmtime_benchmark(iterations: int = 1000) -> dict:
    """真实 wasmtime CLI 跑 N 次 add(1,2) 断言, 返回耗时统计.

    用 wast runner (WAT Script Test format) 而非 --invoke:
      - --invoke 在 wasmtime 45 中标 experimental, 结果不输出到 stdout
      - wast runner 稳定: assert_return 失败 → exit 1, 否则 → exit 0
    """
    wasmtime_path = find_wasmtime()

    # 写 .wast 临时文件: 模块定义 + N 次 add(1,2) 断言
    assertions = '(assert_return (invoke "add" (i32.const 1) (i32.const 2)) (i32.const 3))\n'
    wast_content = WASM_ADD_WAT + (assertions * iterations)

    tmpdir = Path(tempfile.gettempdir()) / "wasmtime_bench"
    tmpdir.mkdir(parents=True, exist_ok=True)
    wast_path = tmpdir / f"bench_{iterations}.wast"
    wast_path.write_text(wast_content, encoding="utf-8")

    # 跑 wasmtime wast, 计时
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [wasmtime_path, "wast", str(wast_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        raise_with_help(
            f"wasmtime 不可执行: {wasmtime_path}",
            "确保 wasmtime 是有效的可执行文件.",
        )
    elapsed_s = time.perf_counter() - start

    if result.returncode != 0:
        raise_with_help(
            f"wasmtime wast 失败 (exit={result.returncode}): {result.stderr[:200]}",
            "检查 .wast 文件语法 / wasmtime 版本 (需 v14+ 支持 wast).",
        )

    elapsed_ms = elapsed_s * 1000
    avg_us = elapsed_s * 1_000_000 / iterations
    return {
        "wasmtime_path": wasmtime_path,
        "iterations": iterations,
        "total_ms": round(elapsed_ms, 2),
        "avg_us_per_call": round(avg_us, 2),
        "wast_file": str(wast_path),
        "wast_size_kb": round(wast_path.stat().st_size / 1024, 1),
    }


# ============================================================
# 3. WebGPU 性能参考 (教学对比, 本地不跑)
# ============================================================
def benchmark_throughput() -> None:
    """Llama-3.2-3B Q4 推理吞吐量对比 (tokens/s, 实测参考)."""
    print("--- 3B Q4 推理吞吐 (tokens/s, 桌面 RTX 3060) ---")
    rows = [
        ("WebGPU (MLC-LLM)", 45, "⭐⭐⭐⭐⭐"),
        ("WebGPU (原生 wgpu)", 30, "⭐⭐⭐⭐"),
        ("WebAssembly (WasmEdge)", 8, "⭐⭐"),
        ("WebAssembly (单线程)", 3, "⭐"),
    ]
    print(f"{'方案':<28} {'tokens/s':<10} {'评级'}")
    print("-" * 55)
    for name, tps, rating in rows:
        print(f"{name:<28} {tps:<10} {rating}")


# ============================================================
# 4. 主流程
# ============================================================
def main() -> None:
    if skip_if_mock("the wasmtime CLI and permission to create a temporary benchmark file"):
        return
    print("=== WebGPU vs WebAssembly 浏览器推理对比 ===\n")
    print_comparison()
    decision_tree()
    benchmark_throughput()

    print("\n=== 真实 wasmtime 跑分 (CLI subprocess) ===")
    result = run_wasmtime_benchmark(iterations=1000)
    print(f"wasmtime 路径: {result['wasmtime_path']}")
    print(f"add(1,2) × {result['iterations']} 次 (wast runner 断言)")
    print(f"  临时 wast 文件: {result['wast_file']} ({result['wast_size_kb']} KB)")
    print(f"  总耗时: {result['total_ms']}ms")
    print(f"  平均: {result['avg_us_per_call']}μs/次")
    print()
    print("WebGPU (浏览器内) 对比需在带 GPU 的浏览器交互运行, 本环境无法跑.")
    print("OK")


if __name__ == "__main__":
    main()
