# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.11.3 SGLang 作为 verl rollout 后端
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: sglang (or fallback mock engine)
# run: python 13_sglang_rollout.py --mock
# expected_runtime: <5s for mock / 需 SGLang + 多 GPU
# expected_output: mock 演示 SGLang 启动 + LoRA 热加载 + 8 路并行 rollout
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.3
# Interview hooks:
#   1. SGLang 相对 vLLM 在 RL 后端上的优势？RadixAttention 与结构化生成？
#   2. engine.load_lora() 热加载的关键意义？GRPO 异步训练如何几乎无停机切换策略？
#   3. Rollout 后端选型矩阵：vLLM / SGLang / TGI / TensorRT-LLM 各自最佳场景？

"""
用 SGLang 作为 verl rollout 后端（配置示意）
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_sglang_rollout():
    """Mock 模式演示"""
    print("[MOCK] SGLang 引擎配置 (4 卡 tensor parallel)")
    print()
    print("""
    import sglang as sgl

    # 1) 启动 SGLang 引擎
    engine = sgl.Engine(
        model_path="meta-llama/Llama-3-8B-Instruct",
        tp_size=4,                     # tensor parallel
        mem_fraction_static=0.85,
        enable_lora=True,
        max_loras_per_batch=8,
    )

    # 2) 在 GRPO 训练 step 中: 先热加载当前 LoRA
    engine.load_lora(lora_path="checkpoints/step_1000/", lora_name="policy")

    # 3) 高并发 rollout: 同 prompt 采样 G 个回答
    prompts = ["解方程 x^2 - 5x + 6 = 0"] * 8
    sampling_params = {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 2048, "n": 1}
    outputs = engine.generate(prompts, sampling_params)
    # 每个 prompt 拿到 1 个回答, 循环 8 次得到 G 个 group 样本
    """)
    print()
    print("=" * 70)
    print("Rollout 后端选型对比")
    print("=" * 70)
    print("""
    后端           速度   灵活性  结构化输出  LoRA热加载  适合场景
    ----------   -----  -----   --------    --------    -----------
    vLLM         ★★★★★  ★★★★   ★★★         ★★★★        通用高速 rollout
    SGLang       ★★★★★  ★★★★★  ★★★★★       ★★★★★       推理 + Agent
    TGI          ★★★★   ★★★    ★★          ★★★         HF 生态
    TensorRT-LLM ★★★★★+ ★★     ★★          ★★          NVIDIA 极致性能
    """)
    print()


def real_sglang_rollout():
    """真实 SGLang 调用（需 sglang 库 + GPU）"""
    try:
        import sglang as sgl
    except ImportError:
        print("未安装 sglang, 请先 pip install sglang")
        return

    engine = sgl.Engine(
        model_path="meta-llama/Llama-3-8B-Instruct",
        tp_size=4,
        mem_fraction_static=0.85,
        enable_lora=True,
        max_loras_per_batch=8,
    )

    engine.load_lora(lora_path="checkpoints/step_1000/", lora_name="policy")

    prompts = ["解方程 x^2 - 5x + 6 = 0"] * 8
    sampling_params = {
        "temperature": 0.9, "top_p": 0.95, "max_new_tokens": 2048, "n": 1,
    }
    outputs = engine.generate(prompts, sampling_params)
    print(f"生成样本数: {len(outputs)}")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_sglang_rollout()
    else:
        real_sglang_rollout()
