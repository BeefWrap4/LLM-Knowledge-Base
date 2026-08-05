# ---
# chapter: 12
# topic: 深度学习与 PyTorch
# topic_id: pytorch.conv2d
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 05_conv2d.py
# expected_runtime: <10s
# expected_output: Conv2d 输出 shape, 参数量
# ---
# See: ../../../12_深度学习与PyTorch.md
#
# Interview hooks:
#  1. 输出尺寸公式: H_out = floor((H_in + 2P - K) / S) + 1, 推导一下?
#  2. 3x3 卷积堆叠两次 vs 单个 5x5 卷积, 感受野相同, 参数与表达力差异?
#  3. 1x1 卷积的常见用途 (降维/升维/跨通道融合)？
import torch
import torch.nn as nn

# 卷积层示例
conv = nn.Conv2d(
    in_channels=3,  # 输入通道（RGB 图像）
    out_channels=64,  # 输出通道（64 个特征图）
    kernel_size=3,  # 3x3 卷积核
    stride=1,  # 步长 1
    padding=1,  # 填充 1，保持尺寸不变
    bias=False,  # 配合 BatchNorm 时通常设 False
)


if __name__ == "__main__":
    # 输入: (batch_size, 3, 224, 224)
    x = torch.randn(8, 3, 224, 224)
    out = conv(x)
    print(f"输出 shape: {out.shape}")  # (8, 64, 224, 224)

    # 计算参数量: out_channels * (in_channels * kernel_h * kernel_w + bias)
    params = 64 * (3 * 3 * 3)  # = 1728
    print(f"参数量: {params:,}")
    print("OK")
