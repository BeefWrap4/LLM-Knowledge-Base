# ---
# chapter: 28
# topic: llama.cpp + GGUF 量化推理 (真实 llama.cpp)
# section: 28.4 llama.cpp 多平台
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: llama-cpp-python
# run: python 03_llama_cpp_gguf_quantization.py
# expected_runtime: 1-3s (loading) + ~10-30 tok/s (推理)
# expected_output: 真实 llama-cpp-python 加载 GGUF + 生成
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.2.1, § 28.4
# Interview hooks:
#   1. GGUF 格式相比 PyTorch .bin 有什么核心优势?
#   2. Q4_K_M 和 Q5_K_M 在端侧 7B 推理时怎么选?
#   3. llama.cpp 的 mmap 加载机制如何实现秒级启动?
"""llama.cpp + GGUF 量化等级对比 + 真实 llama-cpp-python 推理."""

from __future__ import annotations

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import skip_if_mock, skip_unless_enabled

# GGUF 量化类型: 精度 vs 模型大小 vs 适用场景
QUANT_TABLE = [
    ("Q2_K", 2, 2.7, "大", "极致压缩, 质量损失明显, 嵌入式/IoT"),
    ("Q3_K_M", 3, 3.3, "中等", "M = medium, 平衡压缩与质量"),
    ("Q4_0", 4, 3.8, "小", "老式对称量化, 已被 Q4_K 取代"),
    ("Q4_K_M", 4, 4.1, "极小", "⭐ 端侧 7B 黄金标准, 推荐默认"),
    ("Q5_K_M", 5, 4.8, "几乎无", "质量优先, 内存允许时选这个"),
    ("Q6_K", 6, 5.5, "几乎无", "接近 FP16, 高端 PC/工作站"),
    ("Q8_0", 8, 7.2, "无", "几乎无损, 研究/基准测试用"),
    ("F16", 16, 13.5, "无", "半精度, 部署用得少"),
]


def print_quant_table() -> None:
    """打印 7B 模型在不同量化等级下的大小与质量."""
    print(f"{'类型':<10} {'位宽':<5} {'大小GB':<8} {'质量损失':<10} {'推荐场景'}")
    print("-" * 75)
    for name, bits, size_gb, loss, scene in QUANT_TABLE:
        print(f"{name:<10} {bits:<5} {size_gb:<8.1f} {loss:<10} {scene}")


def device_recommendation() -> None:
    """根据设备内存推荐量化等级."""
    print("\n--- 端侧设备推荐 ---")
    devices = [
        ("iPhone 15 Pro", 8, "3B Q4_K_M  (1.8GB)"),
        ("MacBook Air M2", 16, "7B Q4_K_M  (4.1GB)"),
        ("MacBook Pro M3 Max", 64, "70B Q4_K_M (40GB)"),
        ("Snapdragon 8 Gen 3", 12, "7B Q4_K_M  (4.1GB)"),
        ("RTX 4090", 24, "70B Q4_K_M (40GB) - 量化"),
    ]
    print(f"{'设备':<25} {'内存GB':<8} {'推荐配置'}")
    print("-" * 60)
    for dev, mem, rec in devices:
        print(f"{dev:<25} {mem:<8} {rec}")


def main() -> None:
    if skip_if_mock("llama-cpp-python 和本地 GGUF 模型"):
        return
    if skip_unless_enabled(
        "LLAMA_CPP_RUN", "the llama-cpp-python backend and a reviewed local GGUF path"
    ):
        return
    # 1. 量化等级参考表 (无需库, 教学用)
    print_quant_table()
    device_recommendation()
    print()

    # 2. 真实 llama-cpp-python 调用
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except ImportError as e:
        raise_with_help(
            f"无法 import llama_cpp: {e}",
            "运行 `pip install llama-cpp-python`. "
            "Windows 无 GPU 时装 CPU 版即可; macOS/Linux 可选 "
            "`CMAKE_ARGS='-DGGML_METAL=ON' pip install` 编译 GPU 版.",
        )

    model_path = str(_code_root / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf")
    if not Path(model_path).exists():
        raise_with_help(
            f"找不到 GGUF 模型 {model_path}",
            "运行 `make download-models-edge` 下载 GGUF 量化模型, "
            "或手动从 https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF "
            "下载 llama-3.2-3b-instruct-q4_k_m.gguf 到 code/models/ 目录.",
        )

    print(f"加载 GGUF: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # 上下文窗口
        n_threads=4,  # CPU 线程数 (macOS 推荐 = 物理核数)
        n_gpu_layers=0,  # 0=纯CPU; Metal/MPS 用 -1; CUDA 用 -1
        verbose=True,
    )

    # 3. 真实推理调用
    prompt = "讲一个中文冷笑话"
    print(f"\nPrompt: {prompt}")
    response = llm(
        prompt,
        max_tokens=128,
        temperature=0.7,
        stop=["</s>", "\n\n"],
    )
    text = response["choices"][0]["text"]
    print(f"llama.cpp response: {text}")

    # 4. 资源清理
    print("\n✅ GGUF 优势: 单文件 + mmap + 跨平台 + 量化粒度细 (加载时间 ≈ 50ms)")
    print("OK")


if __name__ == "__main__":
    main()
