# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.2.1 Tensor 张量运算
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch, numpy
# run: python 01_tensor_ops.py
# expected_runtime: <5s
# expected_output: Printout of tensor shapes, op results, dtypes
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.2.1-tensor-张量运算
#
# Interview hooks:
#  1. torch.tensor vs torch.from_numpy 的内存共享语义？
#  2. .view() 与 .reshape() 的区别是什么？何时会触发拷贝？
#  3. 解释 PyTorch 的广播机制 (broadcasting) 并举例。
import numpy as np
import torch

# ========== Tensor 创建 ==========
# 从列表创建
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

# 特殊 Tensor
zeros = torch.zeros(3, 4)  # 全零
ones = torch.ones(2, 3)  # 全一
rand = torch.rand(3, 3)  # 均匀分布 [0,1)
randn = torch.randn(2, 3)  # 标准正态分布 N(0,1)
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# GPU 张量（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_gpu = torch.randn(3, 3).to(device)

# ========== 核心运算 ==========
a = torch.randn(2, 3)
b = torch.randn(3, 4)

# 矩阵乘法
c = torch.matmul(a, b)  # 或 a @ b
c = torch.mm(a, b)  # 2D 专用（更快）

# 广播机制
v = torch.randn(3)
a + v  # (2,3) + (3,) → 广播为 (2,3) + (2,3)

# 维度操作
x = torch.randn(4, 5)
x.sum(dim=1)  # 沿列求和，结果 shape (4,)
x.mean(dim=0, keepdim=True)  # 沿行求平均，keepdim=True 保持维度
x_view = x.view(2, 10)  # 重塑形状（共享内存）
x_reshape = x.reshape(2, 10)  # 重塑形状（可能拷贝）
x_unsq = x.unsqueeze(0)  # 在 dim=0 增加维度，(4,5) → (1,4,5)
x_sq = x.squeeze()  # 移除所有 size=1 的维度

# ========== 与 NumPy 互转 ==========
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)  # 共享内存
arr2 = t.numpy()  # 共享内存（CPU Tensor）


if __name__ == "__main__":
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    print(f"zeros shape: {zeros.shape}, ones shape: {ones.shape}")
    print(f"rand sum: {rand.sum().item():.4f}, randn mean: {randn.mean().item():.4f}")
    print(f"arange: {arange.tolist()}")
    print(f"linspace: {linspace.tolist()}")
    print(f"x_gpu device: {x_gpu.device}")
    print(f"a shape: {a.shape}, b shape: {b.shape}, c shape: {c.shape}")
    print(f"a+v shape (broadcast): {(a + v).shape}")
    print(f"x.sum(dim=1) shape: {x.sum(dim=1).shape}")
    print(f"x.mean(dim=0, keepdim=True) shape: {x.mean(dim=0, keepdim=True).shape}")
    print(f"x.view(2,10) shape: {x_view.shape}")
    print(f"x.reshape(2,10) shape: {x_reshape.shape}")
    print(f"x.unsqueeze(0) shape: {x_unsq.shape}")
    print(f"x.squeeze() shape: {x_sq.shape}")
    print(f"t (from numpy) shape: {t.shape}, shares memory: {t.data_ptr() == arr.ctypes.data}")
    print("OK")
