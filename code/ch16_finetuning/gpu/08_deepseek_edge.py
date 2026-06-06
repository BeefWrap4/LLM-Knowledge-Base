# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.6.4 DeepSeek-EE 端侧部署
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: deepseek-ee (or fallback mock)
# run: python 08_deepseek_edge.py --mock
# expected_runtime: <5s for mock
# expected_output: mock 模式打印 DeepSeek-EE 部署选项 + Test-Time Compute 推理演示
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.6.4
# Interview hooks:
#   1. DeepSeek-EE 相对 llama.cpp 的优势？DeepSeek 模型深度优化？
#   2. enable_thinking / thinking_budget 在端侧推理中的作用？
#   3. 不同部署层级的模型选型（端/边/云）如何做权衡？

"""
DeepSeek-EE 端侧部署示例
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_deepseek_ee():
    """Mock 模式演示"""
    print("[MOCK] DeepSeek 部署方案选择")
    print()
    print("  场景             方案                     模型                       硬件")
    print("  ------------  --------------------  ------------------------  -----------------")
    print("  端侧 (手机)     DeepSeek-EE INT4       R1-Distill-1.5B/3B        手机 NPU")
    print("  个人 PC        DeepSeek-EE / llama.cpp R1-Distill-7B/14B       RTX 3060+ / Mac M")
    print("  边缘服务器     vLLM + AWQ             R1-Distill-14B/32B        A10 / L4")
    print("  云端生产       vLLM + TensorRT-LLM    DeepSeek-V3 / R1          A100 / H100")
    print()
    print("[MOCK] DeepSeek-EE 性能指标 (7B INT4)")
    print("  RTX 4090:        ~80 tokens/s")
    print("  MacBook M3 Pro:  ~45 tokens/s")
    print("  骁龙 8 Gen 3:    ~15 tokens/s")
    print()
    print("OK")


def real_deepseek_ee():
    """真实 DeepSeek-EE（需安装 deepseek-ee 库）"""
    try:
        from deepseek_ee import DeepSeekEngine
    except ImportError:
        print("未安装 deepseek-ee, 请先 pip install deepseek-ee")
        print("（本文件 mock 模式可独立运行）")
        return

    # 初始化端侧推理引擎
    engine = DeepSeekEngine(
        model_path="deepseek/DeepSeek-R1-Distill-Qwen-7B",
        device="gpu",            # "cpu" | "gpu" | "npu"
        quantization="int4",     # "fp16" | "int8" | "int4"
        max_batch_size=4,
        max_seq_len=4096,
    )

    # 推理（支持长思考模式 —— Test-Time Compute）
    result = engine.generate(
        prompt="证明: 对于任意正整数 n, n^3 - n 能被 6 整除。",
        max_new_tokens=1024,
        temperature=0.6,
        enable_thinking=True,     # 启用长思考模式
        thinking_budget=512,      # 思考过程最大 token 数
    )

    print(f"思考过程: {result.thinking}")
    print(f"最终答案: {result.text}")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_deepseek_ee()
    else:
        real_deepseek_ee()
