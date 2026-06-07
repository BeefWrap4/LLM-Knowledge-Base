# ---
# chapter: 28
# topic: llama.cpp GGUF 量化与加载
# section: 28.4 llama.cpp 多平台
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: llama-cpp-python
# run: python 03_llama_cpp_gguf_quantization.py
# expected_runtime: <1s (mock mode)
# expected_output: GGUF 量化等级对比 + 模拟 llama.cpp 加载
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.2.1, § 28.4
# Interview hooks:
#   1. GGUF 格式相比 PyTorch .bin 有什么核心优势?
#   2. Q4_K_M 和 Q5_K_M 在端侧 7B 推理时怎么选?
#   3. llama.cpp 的 mmap 加载机制如何实现秒级启动?
"""llama.cpp GGUF 量化等级对比与加载演示 (mock 模式)."""
from __future__ import annotations

# GGUF 量化类型: 精度 vs 模型大小 vs 适用场景
QUANT_TABLE = [
    ("Q2_K",   2,  2.7, "大",     "极致压缩, 质量损失明显, 嵌入式/IoT"),
    ("Q3_K_M", 3,  3.3, "中等",   "M = medium, 平衡压缩与质量"),
    ("Q4_0",   4,  3.8, "小",     "老式对称量化, 已被 Q4_K 取代"),
    ("Q4_K_M", 4,  4.1, "极小",   "⭐ 端侧 7B 黄金标准, 推荐默认"),
    ("Q5_K_M", 5,  4.8, "几乎无", "质量优先, 内存允许时选这个"),
    ("Q6_K",   6,  5.5, "几乎无", "接近 FP16, 高端 PC/工作站"),
    ("Q8_0",   8,  7.2, "无",     "几乎无损, 研究/基准测试用"),
    ("F16",   16,  13.5, "无",    "半精度, 部署用得少"),
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
        ("iPhone 15 Pro",        8,  "3B Q4_K_M  (1.8GB)"),
        ("MacBook Air M2",      16,  "7B Q4_K_M  (4.1GB)"),
        ("MacBook Pro M3 Max",  64,  "70B Q4_K_M (40GB)"),
        ("Snapdragon 8 Gen 3",  12,  "7B Q4_K_M  (4.1GB)"),
        ("RTX 4090",            24,  "70B Q4_K_M (40GB) - 量化"),
    ]
    print(f"{'设备':<25} {'内存GB':<8} {'推荐配置'}")
    print("-" * 60)
    for dev, mem, rec in devices:
        print(f"{dev:<25} {mem:<8} {rec}")


def mock_llama_cpp_load(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 35) -> dict:
    """模拟 llama-cpp-python Llama() 构造. 真实使用需要 GGUF 文件."""
    # 真实代码:
    # from llama_cpp import Llama
    # llm = Llama(
    #     model_path="models/llama-3.2-3b-instruct.Q4_K_M.gguf",
    #     n_ctx=2048,
    #     n_gpu_layers=35,   # GPU 卸载层数, 0=纯CPU, 99=全GPU
    #     n_threads=8,
    # )
    # output = llm("Q: Name the planets\nA:", max_tokens=64, stop=["\n"])
    return {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "loaded": True,
        "backend": "Metal/CUDA/Vulkan/CPU (auto-detect)",
    }


def main() -> None:
    print_quant_table()
    device_recommendation()
    print()
    info = mock_llama_cpp_load("models/llama-3.2-3b-instruct.Q4_K_M.gguf")
    print(f"加载结果: {info}")
    print("\n💡 GGUF 优势: 单文件 + mmap + 跨平台 + 量化粒度细")


if __name__ == "__main__":
    main()
