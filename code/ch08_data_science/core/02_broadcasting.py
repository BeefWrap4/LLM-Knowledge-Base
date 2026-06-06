# ---
# chapter: 8
# topic: 数据科学核心库 - NumPy 广播机制
# section: 8.1.2
# difficulty: 高
# tier: core
# deps: numpy
# run: python 02_broadcasting.py
# expected_runtime: <5s
# expected_output: 广播规则示例输出以及标准化后均值≈0、标准差≈1
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.1.2 广播机制详解)
# Interview hooks:
#   1. shape 为 (5, 4, 3, 2) 与 (3, 2) 能否广播？结果 shape 是什么？
#   2. 简述 NumPy 广播的三条规则（从右往左匹配）。
#   3. 为什么 (3, 4) 与 (5,) 不能广播？

import numpy as np

# ========== 广播规则演示 ==========

# 示例1: 标量 + 数组
arr = np.array([1, 2, 3])
print(arr + 10)  # [11, 12, 13] — 标量广播到 (3,)

# 示例2: (3, 4) + (4,) → (3, 4)
a = np.ones((3, 4))
b = np.arange(4)  # (4,)
print((a + b).shape)  # (3, 4)

# 示例3: (3, 1) + (1, 3) → (3, 3)
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([[10, 20, 30]])   # (1, 3)
print(a + b)
# [[11, 21, 31],
#  [12, 22, 32],
#  [13, 23, 33]]

# 示例4: 失败的情况
a = np.ones((3, 4))
b = np.ones(5)
# a + b  # ValueError: operands could not be broadcast together (3,4) (5,)

# ========== 实际应用：数据标准化 ==========
np.random.seed(42)
data = np.random.randn(100, 5)  # 100 个样本，5 个特征

# 按特征标准化（列方向）
mean = data.mean(axis=0)    # (5,) — 每个特征的均值
std = data.std(axis=0)      # (5,) — 每个特征的标准差

# 广播: (100, 5) - (5,) → (100, 5)
#       (100, 5) / (5,) → (100, 5)
normalized = (data - mean) / std
print(f"标准化后均值: {normalized.mean(axis=0)}")  # ≈ 0
print(f"标准化后标准差: {normalized.std(axis=0)}")  # ≈ 1

# ========== 实际应用：图像处理 ==========
# 给灰度图增加颜色通道权重
image = np.random.rand(256, 256, 3)  # RGB 图像
weights = np.array([0.299, 0.587, 0.114])  # (3,) — RGB 转灰度权重

# 广播: (256, 256, 3) * (3,) → (256, 256, 3) → sum(axis=2) → (256, 256)
grayscale = (image * weights).sum(axis=2)
print(grayscale.shape)  # (256, 256)

if __name__ == "__main__":
    print("OK")
