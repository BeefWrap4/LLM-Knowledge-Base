# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.5.4 混合精度训练 (FP16/BF16)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch (CUDA recommended)
# run: python 10_amp_fp16.py
# expected_runtime: <5s on GPU, CPU 上 autocast 也会跑 (数值按 FP32/BF16 走)
# expected_output: 演示 autocast + GradScaler 用法骨架 (可独立运行)
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.5.4-混合精度训练-fp16bf16
#
# Interview hooks:
#  1. FP16 训练为何需要 Loss Scaling? 缩放因子过大/过小的影响?
#  2. BF16 与 FP16 的指数位/尾数位差异, 为什么 BF16 不用 Loss Scaling?
#  3. autocast 的粒度 (装饰器/with/模块级) 与 dtypes 决策策略?
import torch
from torch.cuda.amp import GradScaler, autocast

# 混合精度训练（PyTorch）
scaler = GradScaler()  # 梯度缩放，防止 FP16 下溢


# 训练循环骨架（无真实数据, 仅展示 API 接入点）
def amp_train_step(model, batch_X, batch_y, criterion, optimizer, scaler, device):
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    optimizer.zero_grad()

    # autocast 自动选择合适的精度
    with autocast():
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

    # 缩放梯度后反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


if __name__ == "__main__":
    # 在 CPU 上跑一遍最小验证, 确保 API 接线正确
    import torch.nn as nn

    model = nn.Linear(20, 3)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = GradScaler()  # CPU 上 GradScaler 仍可调用, 内部 no-op

    bx = torch.randn(4, 20)
    by = torch.randint(0, 3, (4,))
    loss_val = amp_train_step(model, bx, by, crit, opt, sc, torch.device("cpu"))
    print(f"amp_train_step loss (CPU stub): {loss_val:.4f}")
