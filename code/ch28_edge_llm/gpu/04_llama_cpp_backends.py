# ---
# chapter: 28
# topic: llama.cpp 多平台后端 (真实 Llama() 后端选择)
# section: 28.4.1 llama.cpp 后端
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: llama-cpp-python
# run: python 04_llama_cpp_backends.py
# expected_runtime: 1-3s (loading) + 推理时间视 backend 而定
# expected_output: 根据 platform 自动选 Metal/CUDA/CPU backend + 真实 Llama() 调用
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.4.1
# Interview hooks:
#   1. llama.cpp 的 Metal 后端相比 PyTorch MPS 性能如何?
#   2. 为什么 Hexagon NPU 后端只支持部分模型 (受限于算子)?
#   3. 跨平台端侧部署时, 如何选择 backend?
"""演示 llama.cpp 多平台后端选择 + 真实 Llama() 后端配置."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import skip_if_mock, skip_unless_enabled

# llama.cpp 支持的后端矩阵
BACKENDS = [
    ("Metal", "Apple Silicon", "GPU", "macOS / iOS 原生", "✅"),
    ("CUDA", "NVIDIA GPU", "GPU", "Linux/Windows 数据中心", "✅"),
    ("Vulkan", "Cross-platform", "GPU", "Linux/Windows/Android 通用", "✅"),
    ("OpenCL", "Adreno/Mali", "GPU", "Android GPU 兼容路径", "✅"),
    ("Hexagon", "Snapdragon NPU", "NPU", "高通手机/车载", "🆕 2026"),
    ("CANN", "华为昇腾 NPU", "NPU", "Atlas 推理卡", "🆕 2026"),
    ("MUSA", "Moore Threads", "GPU", "国产 GPU 适配", "🆕 2026"),
    ("CPU", "任意 x86/ARM", "CPU", "AVX2/AVX-512/NEON 加速", "✅"),
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
        ("Metal (macOS)", "CMAKE_ARGS='-DGGML_METAL=ON' pip install llama-cpp-python"),
        ("CUDA (Linux/Win)", "CMAKE_ARGS='-DGGML_CUDA=ON' pip install llama-cpp-python"),
        ("Vulkan (Linux)", "CMAKE_ARGS='-DGGML_VULKAN=ON' pip install llama-cpp-python"),
        ("OpenCL (Android)", "cmake -B build -DGGML_OPENCL=ON -DANDROID_ABI=arm64-v8a .."),
        ("Hexagon NPU", "cmake -B build -DGGML_HEXAGON=ON && cmake --build build"),
    ]
    for name, cmd in cmds:
        print(f"  {name}:")
        print(f"    {cmd}")


def _select_backend() -> tuple[str, int]:
    """根据 platform 自动选 backend + n_gpu_layers.

    Returns:
        (backend_name, n_gpu_layers) — n_gpu_layers: 0=纯CPU, -1=全GPU.
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return "Metal", -1  # 全部卸载到 Apple GPU
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        return "CUDA/Vulkan", -1  # 假设有 NVIDIA
    if system == "Windows":
        return "CUDA (有 NVIDIA) / CPU", 0
    if machine == "aarch64" and system == "Linux":
        return "CPU (NEON)", 0
    return "CPU (fallback)", 0


def main() -> None:
    if skip_if_mock("匹配当前平台的 llama.cpp 后端和本地 GGUF 模型"):
        return
    if skip_unless_enabled(
        "LLAMA_CPP_RUN", "the matching llama.cpp backend and a reviewed local GGUF path"
    ):
        return
    # 1. 后端能力矩阵
    print_backend_table()
    compile_commands()

    # 2. 当前硬件自动选 backend
    backend, n_gpu_layers = _select_backend()
    print(f"\n--- 自动后端选择 (系统: {platform.system()} {platform.machine()}) ---")
    print(f"  选中 backend:      {backend}")
    print(f"  n_gpu_layers 配置: {n_gpu_layers}  (0=纯CPU, -1=全GPU, N=前N层GPU)")
    print(f"  理由: {platform.system()} + {platform.machine()} → {backend}")

    # 3. 真实 Llama() 构造 (按选中 backend 配置)
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except ImportError as e:
        raise_with_help(
            f"无法 import llama_cpp: {e}",
            f"运行 `pip install llama-cpp-python`. 当前 backend={backend}, 需匹配编译选项.",
        )

    model_path = str(_code_root / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf")
    if not Path(model_path).exists():
        raise_with_help(
            f"找不到 GGUF 模型 {model_path}",
            "运行 `make download-models-edge` 下载 GGUF 量化模型, "
            "或手动从 https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF "
            "下载 llama-3.2-3b-instruct-q4_k_m.gguf.",
        )

    print(f"\n加载 GGUF ({backend} backend): {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    # 4. 真实推理: 测试 backend 是否生效
    prompt = "Hello, please introduce llama.cpp backends in one sentence."
    response = llm(prompt, max_tokens=64, temperature=0.0)
    text = response["choices"][0]["text"].strip()
    print(f"Response ({backend}): {text}")
    print("\n✅ backend 验证完成 (若 text 正常生成 → backend 工作)")
    print("OK")


if __name__ == "__main__":
    main()
