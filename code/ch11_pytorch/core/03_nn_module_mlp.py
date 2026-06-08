# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.2.3 nn.Module 与模型定义
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 03_nn_module_mlp.py
# expected_runtime: <10s
# expected_output: 模型参数量统计, 一次前向输出 shape
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.2.3-nnmodule-与模型定义
#
# Interview hooks:
#  1. nn.Module 子类化与 nn.Sequential 的取舍? 何时必须自己写 forward?
#  2. He 初始化 (kaiming_normal_) 的数学原理, mode='fan_in'/'fan_out' 的差异?
#  3. 为什么 BatchNorm/Dropout 在 model.eval() 时行为不同?
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 自定义模型 ==========
class MLPClassifier(nn.Module):
    """多层感知机分类器 — 展示标准写法"""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        # 网络层定义
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """He 初始化 — ReLU 激活的标准做法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # 输出层（无激活，CrossEntropyLoss 内部含 Softmax）
        x = self.fc3(x)
        return x


# 实例化模型
model = MLPClassifier(input_dim=784, hidden_dim=256, num_classes=10)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ========== 使用 nn.Sequential 简化 ==========
simple_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)


if __name__ == "__main__":
    x_dummy = torch.randn(4, 784)
    y = model(x_dummy)
    print(f"MLPClassifier output shape: {y.shape}")
    y2 = simple_model(x_dummy)
    print(f"nn.Sequential model output shape: {y2.shape}")
