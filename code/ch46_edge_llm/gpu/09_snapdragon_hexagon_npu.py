# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: edge_llm.snapdragon_hexagon_npu
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: 外部工具链（Snapdragon 设备、Android NDK、Hexagon SDK、llama.cpp）
# run: python 09_snapdragon_hexagon_npu.py
# expected_runtime: <1s (structural skip)
# expected_output: current official prerequisites and build/run outline, then SKIP + OK
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
# Official source:
# https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md
# Interview hooks:
# 1. Hexagon 后端为何仍需 CPU 参与调度，不能把 NPU TOPS 直接换算为 token/s？
# 2. prefill 与 decode 对算力、内存带宽和 KV cache 的瓶颈有何不同？
# 3. 怎样证明一次推理确实把目标算子卸载到了 HTP，而不是回退到 CPU/OpenCL？
"""Snapdragon Hexagon 后端的诚实边界与当前官方命令。

本文件不含 QNN/Hexagon Python 绑定，也不会伪造 NPU 推理。真实验收必须在
Snapdragon 设备上按 llama.cpp 当前 ``docs/backend/snapdragon`` 文档构建，
并从运行日志确认 HTP 设备、offload 层数、算子回退和性能指标。
"""


def print_capability_boundary() -> None:
    print("=== llama.cpp Snapdragon / Hexagon 后端 ===")
    print("状态: 官方仓库中的 experimental backend，接口与算子覆盖仍可能变化")
    print("平台: 文档覆盖 Snapdragon Android；另列 Windows on Snapdragon 原生构建条件")
    print("设备: HTP0、HTP1 等 Hexagon session；模型/量化/算子支持以当前 README 为准")
    print("性能: TOPS、CPU/GPU/NPU 利用率不能替代同模型同参数的 pp/tg 实测")


def show_current_build_outline() -> None:
    print("\n--- 官方 Android 工具链路径（命令提纲）---")
    print("1. 使用 llama.cpp 文档指定的 snapdragon-toolchain 容器与版本")
    print("2. 在仓库根目录复制并使用官方 CMake preset:")
    print("   cp docs/backend/snapdragon/CMakeUserPresets.json .")
    print("   cmake --preset arm64-android-snapdragon-release -B build-snapdragon")
    print("   cmake --build build-snapdragon")
    print("3. 使用仓库 wrapper 设置运行库、ADB 与 HTP 设备，例如:")
    print(
        '   M=Llama-3.2-1B-Instruct-Q4_0.gguf D=HTP0 '
        './scripts/snapdragon/adb/run-completion.sh -p "Hello"'
    )
    print("4. 验收日志: Hexagon arch、HTP session、offloaded layers、回退算子、pp/tg token/s")
    print("提示: 镜像版本、SDK 版本、preset 名和支持模型均应在执行当天复核官方 README")


def main() -> None:
    print_capability_boundary()
    show_current_build_outline()
    print("\n[SKIP] 本 Python 文件仅做结构验收；真实 Hexagon 推理需 Snapdragon 设备和外部工具链")
    print("OK")


if __name__ == "__main__":
    main()
