# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.1 多模态 LoRA - 参数注入教学骨架
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 06_lora_finetune.py
# expected_runtime: <5s
# expected_output: LoRA 分支的可训练参数与输出形状
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-1-多模态lora微调
# Interview hooks:
#   1. 冻结视觉编码器、训练 projector 与给 LLM 注入 LoRA 的取舍是什么？
#   2. LoRA 的 rank、alpha 与 target modules 如何影响容量和资源占用？
#   3. 为什么真实多模态微调必须核对 checkpoint 的模块名和 processor？

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """最小 LoRA 线性层；用于解释公式，不替代 PEFT。"""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = (inputs @ self.lora_a.T) @ self.lora_b.T
        return self.base(inputs) + update * self.scale


def main() -> None:
    """演示 LoRA 参数注入和 projector 训练边界，不加载或训练真实 VLM。"""
    torch.manual_seed(42)
    layer = LoRALinear(in_features=64, out_features=64, rank=4, alpha=8)
    projector = nn.Linear(32, 64)
    inputs = torch.randn(2, 8, 64)
    projected_vision = projector(torch.randn(2, 4, 32))
    outputs = layer(inputs)

    trainable = sum(
        parameter.numel()
        for module in (layer, projector)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    assert not layer.base.weight.requires_grad
    assert outputs.shape == inputs.shape
    assert projected_vision.shape == (2, 4, 64)
    print("[STRUCTURE ONLY] No VLM checkpoint, quantization, dataset, optimizer, or training run.")
    print(f"trainable_parameters={trainable}, output={tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
    print("OK")
