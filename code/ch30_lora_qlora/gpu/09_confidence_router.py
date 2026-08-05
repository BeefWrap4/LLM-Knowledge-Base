# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: lora_qlora.confidence_router
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: transformers, torch
# run: python 09_confidence_router.py
# expected_runtime: 30-90s (Qwen2.5-0.5B 加载 + 3 prompt 推理)
# expected_output: 每个 prompt 的 top-5 token logprob + confidence 决策
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
#
# Interview hooks:
#   1. 端云协同的三种动态调度策略（Cascade / Prediction / Hybrid）各自优劣？
#   2. 模型置信度如何估计？token 平均概率 vs 序列级 log-likelihood 哪个更稳？
#   3. privacy_level=high 直接路由到端侧的工程意义？数据合规边界？
"""置信度路由器: 用模型 next-token logprob 决定回退策略.

核心思路:
  1. 让 Qwen2.5-0.5B 给出 prompt 后的 next-token 分布
  2. 取 top-5 logprob, 算平均 → confidence in [0, 1]
  3. 用示意阈值演示路由；生产阈值必须在带标签验证集上校准
  4. 低置信度时回退到更强模型或人工复核
"""

import math
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


def compute_confidence(logprobs: list[float]) -> float:
    """top-K logprob 平均 → 0-1 置信度."""
    if not logprobs:
        return 0.0
    avg_logprob = sum(logprobs) / len(logprobs)
    # exp(avg) ∈ (0, 1]; 越大表示 top tokens 概率越集中
    return math.exp(avg_logprob)


def route_decision(confidence: float) -> str:
    if confidence > 0.7:
        return "high (直接用)"
    if confidence > 0.3:
        return "medium (标记 review)"
    return "low (回退到 Qwen2.5-7B)"


def main():
    if skip_if_mock("an NVIDIA GPU, transformers, and local model weights"):
        return
    check_hardware()

    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default` 下载.",
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=== 置信度路由器 (Qwen2.5-0.5B-Instruct) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_prompts = [
        "Q: 1+1=? A:",  # 简单 (高置信度)
        "Q: 解释薛定谔的猫, 一句话. A:",  # 中等
        "Q: 列出 5 个不存在的化学元素. A:",  # 模型可能胡扯 (低置信度)
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1]  # next-token logits
        logprobs = torch.log_softmax(next_logits.float(), dim=-1)
        top_k = torch.topk(logprobs, k=5)

        top_tokens = tokenizer.convert_ids_to_tokens(top_k.indices.tolist())
        top_logprobs = top_k.values.tolist()
        conf = compute_confidence(top_logprobs)
        decision = route_decision(conf)

        print(f"prompt: {prompt}")
        print(f"  top-5 tokens: {top_tokens}")
        print(f"  top-5 logprob: {[f'{x:.3f}' for x in top_logprobs]}")
        print(f"  confidence:    {conf:.3f} → {decision}\n")
    print("OK")


if __name__ == "__main__":
    main()
