# ---
# chapter: 28
# topic: llama.cpp 多平台后端 (Metal/CUDA/Vulkan/Hexagon)
# section: 28.4.1 llama.cpp 后端
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: llama-cpp-python
# run: python 04_llama_cpp_backends.py
# expected_runtime: <1s
# expected_output: 后端能力矩阵 + 编译参数演示
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.4.1
# Interview hooks:
#   1. llama.cpp 的 Metal 后端相比 PyTorch MPS 性能如何?
#   2. 为什么 Hexagon NPU 后端只支持部分模型 (受限于算子)?
#   3. 跨平台端侧部署时, 如何选择 backend?
"""演示 llama.cpp 多平台后端选择与编译配置."""
from __future__ import annotations

# llama.cpp 支持的后端矩阵
BACKENDS = [
    ("Metal",    "Apple Silicon",  "GPU", "macOS / iOS 原生",           "✅"),
    ("CUDA",     "NVIDIA GPU",     "GPU", "Linux/Windows 数据中心",     "✅"),
    ("Vulkan",   "Cross-platform", "GPU", "Linux/Windows/Android 通用", "✅"),
    ("OpenCL",   "Adreno/Mali",    "GPU", "Android GPU 兼容路径",        "✅"),
    ("Hexagon",  "Snapdragon NPU", "NPU", "高通手机/车载",              "🆕 2026"),
    ("CANN",     "华为昇腾 NPU",   "NPU", "Atlas 推理卡",               "🆕 2026"),
    ("MUSA",     "Moore Threads",  "GPU", "国产 GPU 适配",              "🆕 2026"),
    ("CPU",      "任意 x86/ARM",   "CPU", "AVX2/AVX-512/NEON 加速",    "✅"),
]


def print_backend_table() -> None:
    print(f"{'后端':<10} {'硬件':<20} {'类型':<5} {'场景':<30} {'状态'}")
    print("-" * 80)
    for name, hw, kind, scene, status in BACKENDS:
        print(f"{name:<10} {hw:<20} {kind:<5} {scene:<30} {status}")


def compile_commands() -> None:
    """演示 llama.cpp 的编译选项 - 不同后端."""
    print("\n--- 编译选项 (CMake) ---")
    cmds = [
        ("Metal (macOS)",    "cmake -B build -DGGML_METAL=ON && cmake --build build"),
        ("CUDA (Linux)",     "cmake -B build -DGGML_CUDA=ON && cmake --build build"),
        ("Vulkan (Linux)",   "cmake -B build -DGGML_VULKAN=ON && cmake --build build"),
        ("OpenCL (Android)", "cmake -B build -DGGML_OPENCL=ON -DANDROID_ABI=arm64-v8a .."),
        ("Hexagon NPU",      "cmake -B build -DGGML_HEXAGON=ON && cmake --build build"),
    ]
    for name, cmd in cmds:
        print(f"  {name}:")
        print(f"    {cmd}")


def mock_runtime_selection() -> None:
    """模拟运行时根据硬件选择 backend."""
    import platform
    system = platform.system()
    machine = platform.machine()
    print(f"\n--- 自动后端选择 (系统: {system} {machine}) ---")

    if system == "Darwin" and machine == "arm64":
        backend = "Metal"
        reason = "Apple Silicon -> Metal 最高效"
    elif "nvidia" in machine.lower() or system == "Linux":
        backend = "CUDA"
        reason = "NVIDIA GPU 优先 CUDA, 备选 Vulkan"
    elif machine == "aarch64" and system == "Linux":
        backend = "CPU (NEON)"
        reason = "ARM 设备默认 CPU, 可选 OpenCL"
    else:
        backend = "CPU"
        reason = "未知平台, fallback 到 CPU"

    print(f"  选中: {backend}")
    print(f"  理由: {reason}")


def main() -> None:
    print_backend_table()
    compile_commands()
    mock_runtime_selection()


if __name__ == "__main__":
    main()
