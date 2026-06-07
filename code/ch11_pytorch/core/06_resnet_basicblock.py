# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.3.2 经典架构演进 (ResNet BasicBlock)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 06_resnet_basicblock.py
# expected_runtime: <10s
# expected_output: BasicBlock 输出 shape (相同/下采样)
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.3.2-经典架构演进
#
# Interview hooks:
#  1. 残差连接 y = F(x) + x 为何能缓解梯度消失? 数学上如何证明?
#  2. 当维度不匹配时, 1x1 卷积 shortcut 的作用, 能否用 zero-padding 替代?
#  3. Pre-activation ResNet 与 Post-activation ResNet 的训练稳定性差异?
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet 基础残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 下采样层（当维度不匹配时）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out


if __name__ == "__main__":
    # 相同维度
    block_same = BasicBlock(64, 64, stride=1)
    x_same = torch.randn(2, 64, 56, 56)
    y_same = block_same(x_same)
    print(f"same-dim  out shape: {y_same.shape}")
    # 维度不匹配 (下采样 + 通道翻倍)
    block_down = BasicBlock(64, 128, stride=2)
    x_down = torch.randn(2, 64, 56, 56)
    y_down = block_down(x_down)
    print(f"downsample out shape: {y_down.shape}")
