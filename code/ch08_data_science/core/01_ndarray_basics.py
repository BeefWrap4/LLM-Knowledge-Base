# ---
# chapter: 8
# topic: 数据科学核心库 - NumPy ndarray 核心概念
# section: 8.1.1
# difficulty: 中高
# tier: core
# deps: numpy
# run: python 01_ndarray_basics.py
# expected_runtime: <5s
# expected_output: ndim/shape/size/dtype/itemsize/nbytes/strides 全部打印
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.1.1 ndarray 核心概念)
# Interview hooks:
#   1. ndarray 与 Python List 在内存布局和性能上有何本质区别？
#   2. strides 在 ndarray 中的作用是什么？为什么说切片是"视图"而非"拷贝"？
#   3. 解释 ndim / shape / dtype / size / nbytes / strides 的含义及相互关系。

import numpy as np

# ========== 创建 ndarray ==========
# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])

# 指定 dtype
arr2 = np.array([1, 2, 3], dtype=np.float32)

# 常用创建函数
zeros = np.zeros((3, 4))  # 3x4 零矩阵
ones = np.ones((2, 3))  # 2x3 全1矩阵
eye = np.eye(3)  # 3x3 单位矩阵
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

# 随机数组
np.random.seed(42)
random_arr = np.random.randn(3, 3)  # 标准正态分布
uniform = np.random.rand(3, 3)  # [0, 1) 均匀分布
randint = np.random.randint(0, 10, (3, 3))  # 随机整数

# ========== 关键属性 ==========
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"ndim: {arr.ndim}")  # 2（维度数）
print(f"shape: {arr.shape}")  # (2, 3)（各维度大小）
print(f"size: {arr.size}")  # 6（元素总数）
print(f"dtype: {arr.dtype}")  # int64（元素类型）
print(f"itemsize: {arr.itemsize}")  # 8（每个元素字节数）
print(f"nbytes: {arr.nbytes}")  # 48（总字节数）
print(f"strides: {arr.strides}")  # (24, 8)（每维步长）

if __name__ == "__main__":
    print("OK")
