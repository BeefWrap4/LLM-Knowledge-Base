# ---
# chapter: 28
# topic: WebGPU vs WebAssembly 浏览器推理对比
# section: 28.5.2 WebGPU vs WebAssembly
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (浏览器侧)
# run: python 08_webgpu_vs_wasm.py
# expected_runtime: <1s
# expected_output: 对比矩阵 + 决策流程
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.5.2
# Interview hooks:
#   1. WebGPU 和 WebAssembly 各自最适合什么场景?
#   2. WebGPU 为什么有 4-8GB 显存限制 (来自 WebGL/Vulkan 抽象层)?
#   3. 量化等级对 WebGPU 性能的影响?
"""WebGPU vs WebAssembly 浏览器推理对比 + 决策树."""
from __future__ import annotations

COMPARISON_MATRIX = [
    ("性能",         "⭐⭐⭐⭐⭐", "⭐⭐⭐",     "WebGPU 利用 GPU 并行"),
    ("浏览器兼容",   "Chromium/Safari", "全部浏览器", "WASM 通用性更强"),
    ("计算单元",     "GPU (并行)",   "CPU (串行)",   "本质区别"),
    ("API 风格",     "现代 (wgpu)",  "低级 (LLVM IR)", "WebGPU 抽象更友好"),
    ("内存",         "GPU 显存 4-8GB", "系统 RAM",   "WebGPU 受限于安全沙箱"),
    ("适用模型大小", "1B-7B 量化",   "<1B",         "WASM 算力天花板"),
    ("功耗",         "中-高 (GPU)",   "低 (CPU)",     "WASM 适合移动端续航"),
    ("首推框架",     "WebLLM/MLC-LLM", "WasmEdge/llama.cpp", "生态成熟度"),
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


def benchmark_throughput() -> None:
    """Llama-3.2-3B Q4 推理吞吐量对比 (tokens/s)."""
    print("--- 3B Q4 推理吞吐 (tokens/s, 桌面 RTX 3060) ---")
    rows = [
        ("WebGPU (MLC-LLM)",       45, "⭐⭐⭐⭐⭐"),
        ("WebGPU (原生 wgpu)",     30, "⭐⭐⭐⭐"),
        ("WebAssembly (WasmEdge)",  8, "⭐⭐"),
        ("WebAssembly (单线程)",    3, "⭐"),
    ]
    print(f"{'方案':<28} {'tokens/s':<10} {'评级'}")
    print("-" * 55)
    for name, tps, rating in rows:
        print(f"{name:<28} {tps:<10} {rating}")


def main() -> None:
    print_comparison()
    decision_tree()
    benchmark_throughput()


if __name__ == "__main__":
    main()
    print("OK")
