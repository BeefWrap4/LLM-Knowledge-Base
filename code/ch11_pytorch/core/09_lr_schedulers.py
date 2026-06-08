# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.5.2 学习率调度
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 09_lr_schedulers.py
# expected_runtime: <5s
# expected_output: 5 步学习率数值, 展示不同 scheduler 行为
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.5.2-学习率调度
#
# Interview hooks:
#  1. ReduceLROnPlateau 与其他 scheduler 在 step() 调用上有什么不同? (需要传 metric)
#  2. CosineAnnealingWarmRestarts 的 T_0/T_mult 含义? 与 OneCycleLR 的对比?
#  3. Warmup 阶段的作用是什么? 为什么 Transformer 训练必须有 warmup?
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# 占位: 实际调度器需要一个 model 才能构建 optimizer; 这里以简单 Linear 演示
_dummy_model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(_dummy_model.parameters(), lr=1e-3)

# 方式1: Step Decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 方式2: Reduce on Plateau（推荐通用场景）
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

# 方式3: Cosine Annealing with Warmup（Transformer 训练）
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


if __name__ == "__main__":
    # 展示 scheduler 对 lr 的影响 (用 fresh optimizer)
    import torch.nn as nn

    m = nn.Linear(10, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = StepLR(opt, step_size=2, gamma=0.5)  # 短 step 便于演示

    print("StepLR lr 走势 (step_size=2, gamma=0.5):")
    for step in range(5):
        print(f"  step={step}  lr={opt.param_groups[0]['lr']:.6f}")
        sch.step()
