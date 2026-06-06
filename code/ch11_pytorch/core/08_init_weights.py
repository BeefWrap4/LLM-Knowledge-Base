# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.5.1 权重初始化
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 08_init_weights.py
# expected_runtime: <5s
# expected_output: 初始化前/后权重统计
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.5.1-权重初始化
#
# Interview hooks:
#  1. Xavier 初始化 vs He 初始化的方差推导差异? 各适合哪种激活函数?
#  2. mode='fan_in' 与 'fan_out' 的区别? Conv2d 为何通常用 fan_out?
#  3. 正交初始化 (orthogonal) 为什么对 RNN 特别有效?
import torch
import torch.nn as nn

# PyTorch 自动初始化
# nn.Linear 默认使用 Kaiming Uniform (for ReLU)
# nn.Conv2d 默认使用 Kaiming Uniform

# 手动初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Linear(16 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    # 初始化前
    w_before = model[0].weight.clone()
    print(f"conv 初始化前 mean={w_before.mean().item():.5f} std={w_before.std().item():.5f}")

    model.apply(init_weights)
    # 初始化后
    w_after = model[0].weight
    print(f"conv 初始化后 mean={w_after.mean().item():.5f} std={w_after.std().item():.5f}")
    print("OK")
