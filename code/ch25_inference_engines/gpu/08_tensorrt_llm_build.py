# ---
# chapter: 25
# topic: TensorRT-LLM Engine Build + Serve
# section: 25.2.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: trtllm-build, trtllm-serve (NVIDIA TensorRT-LLM CLI; Linux only)
# run: python 08_tensorrt_llm_build.py
# expected_runtime: 30s+ (CLI lookup + build dispatched)
# expected_output: 真实 trtllm-build / trtllm-serve 调用; 缺 CLI 时友好抛错
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.3
# Interview hooks:
#   1. TensorRT-LLM 和 vLLM 的核心权衡？(答: TRT 编译慢但运行极快; vLLM 启动快, 迭代友好)
#   2. In-flight batching 是什么？(答: decode 阶段动态插入/驱逐, 类似 continuous batching)
#   3. kernel autotune 在 build 时做什么？(答: 在目标硬件上选最快的 GEMM/attention kernel)

"""TensorRT-LLM Engine Build + Serve 演示 (真实 trtllm-build CLI).

TensorRT-LLM 是 NVIDIA 的 LLM 推理编译器:
  - 把 HuggingFace 模型编译成 TensorRT engine (graph capture + kernel autotune)
  - 通过图优化、量化与目标硬件 kernel 调优换取运行时性能；收益需按基线实测

工作流:
  1. trtllm-build --checkpoint_dir ... --output_dir engine
  2. trtllm-serve engine/
  3. 客户端用 OpenAI 协议访问 (base_url=http://localhost:8000/v1)

注意: 官方 TensorRT-LLM 在 Windows 上**不可用**; 缺 CLI 时本脚本会
通过 `shared._error_helper.raise_with_help` 输出明确安装指引.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help  # noqa: E402
from shared.gpu_guard import (  # noqa: E402
    require_nvidia_gpu,
    skip_if_mock,
    skip_unless_enabled,
)


def check_hardware() -> None:
    """需要 NVIDIA GPU + ≥24GB VRAM (RTX 4090/5090/H100)."""
    require_nvidia_gpu(min_vram_gb=24)


def find_trtllm_build() -> str:
    """查找 trtllm-build CLI 路径. 找不到时抛友好错."""
    path = shutil.which("trtllm-build")
    if path:
        return path
    # 常见安装位置
    candidates = [
        "/usr/local/bin/trtllm-build",
        "/opt/tensorrtllm/bin/trtllm-build",
        str(Path.home() / "tensorrt-llm" / "build" / "trtllm-build"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise_with_help(
        "trtllm-build CLI 未找到",
        "TensorRT-LLM 安装路径: "
        "1) pip install tensorrt-llm (Linux only); "
        "2) Docker nvcr.io/nvidia/tensorrt-llm/release; "
        "3) 从源码编译: https://github.com/NVIDIA/TensorRT-LLM. "
        "当前 Windows 平台官方不支持 (Linux only).",
    )


def find_trtllm_serve() -> str:
    """查找 trtllm-serve CLI 路径. 找不到时抛友好错."""
    path = shutil.which("trtllm-serve")
    if path:
        return path
    raise_with_help(
        "trtllm-serve CLI 未找到",
        "随 TensorRT-LLM 一起安装. 见上方 trtllm-build 安装路径.",
    )


def build_engine(
    checkpoint_dir: str,
    output_dir: str,
    max_batch_size: int = 64,
    max_seq_len: int = 8192,
    tp_size: int = 1,
    timeout_s: int = 3600,
) -> None:
    """真实 trtllm-build 调用.

    Args:
        checkpoint_dir: HF 转 TensorRT-LLM checkpoint 后的目录
        output_dir: 输出 engine 目录
        max_batch_size: 最大 batch size
        max_seq_len: 最大序列长度
        tp_size: tensor parallel size
        timeout_s: 编译超时 (秒, 默认 1h)
    """
    trtllm_build = find_trtllm_build()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        trtllm_build,
        "--checkpoint_dir",
        checkpoint_dir,
        "--output_dir",
        output_dir,
        "--max_batch_size",
        str(max_batch_size),
        "--max_seq_len",
        str(max_seq_len),
        "--gemm_plugin",
        "fp16",
        "--attention_plugin",
        "trtllm",
        "--tp_size",
        str(tp_size),
    ]

    print(f"[trtllm-build] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError as e:
        raise_with_help(f"trtllm-build 不可执行: {e}", "见上方安装说明.")
    except subprocess.TimeoutExpired:
        raise_with_help(
            f"trtllm-build 超时 (>{timeout_s}s)",
            "编译大模型需数小时, 考虑减小 max_batch_size 或换更小的模型.",
        )

    if result.returncode != 0:
        print(f"[stdout] {result.stdout[-500:]}")
        print(f"[stderr] {result.stderr[-500:]}")
        raise_with_help(
            f"trtllm-build 失败 (rc={result.returncode})",
            "检查 checkpoint_dir 路径, TensorRT 版本兼容性, 显存是否充足.",
        )
    print(f"✅ Engine built: {output_dir}")


def main() -> None:
    if skip_if_mock("Linux, an NVIDIA GPU, TensorRT-LLM CLIs, and a local checkpoint"):
        return
    if skip_unless_enabled(
        "TRTLLM_BUILD_RUN", "the TensorRT-LLM toolchain, output path, and local checkpoint"
    ):
        return
    check_hardware()

    # 1. 找 HuggingFace checkpoint
    checkpoint_dir = _code_root / "models" / "Qwen2.5-0.5B-Instruct"
    if not checkpoint_dir.exists():
        raise_with_help(
            f"需要 HF checkpoint: {checkpoint_dir}",
            "运行 `make download-models-default`.",
        )

    # 2. 输出 engine 目录
    output_dir = _code_root / "models" / "Qwen2.5-0.5B-trtllm-engine"

    # 3. Build (单 GPU, 小模型, 快速演示参数)
    build_engine(
        checkpoint_dir=str(checkpoint_dir),
        output_dir=str(output_dir),
        max_batch_size=8,
        max_seq_len=2048,
        tp_size=1,
    )

    # 4. 提示下一步
    trtllm_serve = find_trtllm_serve()
    print(f"\n下一步: {trtllm_serve} {output_dir}/")
    print("然后用 OpenAI 协议客户端: base_url=http://localhost:8000/v1")
    print("示例:")
    print("  curl http://localhost:8000/v1/chat/completions \\")
    print("    -H 'Content-Type: application/json' \\")
    print('    -d \'{"model":"qwen","messages":[{"role":"user","content":"hi"}]}\'')


if __name__ == "__main__":
    main()
