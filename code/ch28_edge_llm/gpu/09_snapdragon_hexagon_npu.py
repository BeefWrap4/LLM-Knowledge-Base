# ---
# chapter: 28
# topic: Snapdragon Hexagon NPU 推理 (设备不可得, 教学展示)
# section: 28.4 llama.cpp + Hexagon 后端
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (Qualcomm AI Engine SDK, 仅 Android 真机可跑, 本地无设备)
# run: python 09_snapdragon_hexagon_npu.py
# expected_runtime: <1s (抛错 + 教学展示)
# expected_output: 明确提示 "需要 Snapdragon NPU" + QNN SDK 安装/使用命令
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.4.1, § 28.6 NPU 后端
# Interview hooks:
#   1. Hexagon NPU 相比 Adreno GPU 推理 LLM 有什么优势?
#   2. 为什么端侧 LLM 量化到 INT4/INT8 很重要?
#   3. Snapdragon 8 Gen 3 跑 7B Q4 模型实际能到多少 tokens/s?
"""Snapdragon Hexagon NPU 端侧 LLM 推理 — 设备不可得, 教学展示.

本环境 (Windows + x86/x64 CPU + NVIDIA RTX 5090D) 无 Snapdragon NPU
设备, 也无 QNN SDK. 文件主要功能:
  1. 调用 check_hardware() 抛友好错 (明确说明需要 Snapdragon NPU)
  2. except 块展示 QNN SDK 安装与使用命令 (教学保留)
  3. 关键 NPU 硬件规格 / NPU vs GPU 权衡 / 量化推荐 (静态打印)
"""
from __future__ import annotations

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help  # noqa: E402


# ============================================================
# 1. Hexagon NPU 关键硬件特性 (Snapdragon 8 Gen 3 / X Elite)
# ============================================================
HEXAGON_SPECS = {
    "峰值 TOPS (INT8)":      "45 TOPS",
    "内存带宽":              "LPDDR5X 8533 MT/s, 77 GB/s",
    "支持精度":              "INT4 / INT8 / INT16 / FP16",
    "最佳 batch":            "batch_size=1 (实时推理)",
    "上下文长度上限":        "受限于 SRAM (约 16MB, ~4K tokens for 7B)",
}


def print_npu_specs() -> None:
    print("--- Snapdragon 8 Gen 3 / X Elite Hexagon NPU ---")
    for k, v in HEXAGON_SPECS.items():
        print(f"  {k}: {v}")


def npu_vs_gpu_tradeoff() -> None:
    """NPU vs GPU 推理对比."""
    print("\n--- NPU vs GPU 推理权衡 ---")
    rows = [
        ("算子覆盖",   "专用矩阵加速 (INT8/INT4)",  "通用 (FP16/FP32)"),
        ("能效比",     "⭐⭐⭐⭐⭐ (~5x GPU)",      "⭐⭐⭐"),
        ("性能峰值",   "中等 (受 SRAM 限制)",         "高 (DDR 高带宽)"),
        ("适用精度",   "INT4/INT8 (必须量化)",         "FP16 (量化可选)"),
        ("LLM 推理",   "✅ 7B Q4 实时",                "✅ 7B Q4 高吞吐"),
        ("训练",       "❌ 不支持反向传播",             "✅ 通用训练"),
    ]
    print(f"{'维度':<12} {'NPU (Hexagon)':<35} {'GPU (Adreno)'}")
    print("-" * 80)
    for dim, npu, gpu in rows:
        print(f"{dim:<12} {npu:<35} {gpu}")


def quantization_recommendation() -> None:
    """Hexagon NPU 上推荐的量化等级."""
    print("\n--- 量化等级推荐 ---")
    print("  INT4 权重量化 (Q4_K_M): 最佳能效比, 7B 模型 4GB → 1.8GB")
    print("  INT8 权重量化 (Q8_0):   质量优先, 适合小模型 (1-3B)")
    print("  FP16 权重:               仅基准测试, 不适合真机部署")
    print("  AWQ/GPTQ INT4:           需要后量化校准, 适合自定义模型")


# ============================================================
# 2. QNN SDK 安装与使用 (教学保留)
# ============================================================
def show_qnn_sdk_install() -> None:
    """展示 QNN SDK 安装与使用命令 (教学保留)."""
    print("Snapdragon NPU 推理需要:")
    print()
    print("  1. Qualcomm AI Engine Direct SDK (QNN SDK)")
    print("     https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk")
    print()
    print("  2. Hexagon DSP 工具链")
    print("     https://developer.qualcomm.com/hexagon-sdk")
    print()
    print("  3. ONNX → QNN 转换:")
    print("     $ qnn-onnx-converter \\")
    print("         --input_network model.onnx \\")
    print("         --output_path model.qnn \\")
    print("         --target_runtime hexagon")
    print()
    print("  4. 在 Snapdragon 设备上加载:")
    print("     $ adb push model.qnn /data/local/tmp/")
    print("     $ adb shell qnn-net-run \\")
    print("         --model /data/local/tmp/model.qnn \\")
    print("         --input /data/local/tmp/input.raw")
    print()
    print("  5. Python 端 SDK 不可用, 需 C++ 绑定.")
    print()
    print("替代方案: 用 llama.cpp 的 Hexagon backend 加载预编译模型")
    print("  $ cmake -B build -DGGML_HEXAGON=ON")
    print("  $ cmake --build build --config Release")
    print("  $ ./build/bin/llama-cli -m model.Q4_K.hexagon.dlc -p \"Hello\"")


# ============================================================
# 3. 硬件检查
# ============================================================
def check_hardware() -> None:
    """设备不可得 → 抛错 + 引导至教学展示."""
    raise_with_help(
        "此例子需要 Snapdragon NPU 设备 (Hexagon DSP).",
        "QNN SDK 仅在 Qualcomm 设备上可用. "
        "本机 (Windows + x86 CPU / NVIDIA GPU) 无 Hexagon NPU. "
        "详见 README §硬件 × 章节矩阵 或 Ch28 §28.4 教程.",
    )


# ============================================================
# 4. 主流程: 抛错 → 教学展示
# ============================================================
def main() -> None:
    # 1) 打印 NPU 静态信息 (即使无设备也展示规格 + 决策依据)
    print_npu_specs()
    npu_vs_gpu_tradeoff()
    quantization_recommendation()
    print()
    # 2) 检查硬件: 抛错
    check_hardware()
    # 实际不会到这里 (check_hardware 抛错)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(str(e))
        print()
        print("--- 教学展示: QNN SDK 安装与使用 ---")
        show_qnn_sdk_install()
