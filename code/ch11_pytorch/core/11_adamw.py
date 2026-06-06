# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.5.5 优化器选择指南 (AdamW)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 11_adamw.py
# expected_runtime: <5s
# expected_output: AdamW 构造成功, 默认超参打印
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.5.5-优化器选择指南
#
# Interview hooks:
#  1. AdamW 中 weight_decay 与 L2 正则的区别, 推导解耦后的更新公式?
#  2. AdamW 的 betas=(0.9, 0.999) 与 eps=1e-8 的作用, 何时需要调整?
#  3. 为什么大模型预训练 (BERT/GPT) 统一使用 AdamW + Warmup + LayerNorm?
import torch
import torch.nn as nn

# AdamW — Transformer 训练的标准配置
model = nn.Linear(10, 2)  # 占位模型
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,           # 学习率（BERT-base 通常 2e-5 ~ 5e-5）
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # 权重衰减系数
)


if __name__ == "__main__":
    print(f"optimizer type: {type(optimizer).__name__}")
    for i, g in enumerate(optimizer.param_groups):
        print(f"group {i}: lr={g['lr']}, betas={g['betas']}, "
              f"eps={g['eps']}, weight_decay={g['weight_decay']}")
    print("OK")
