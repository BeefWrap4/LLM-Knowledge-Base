# ---
# chapter: 8
# topic: 数据科学核心库 - NumPy 矩阵运算与线性代数
# section: 8.1.3
# difficulty: 中高
# tier: core
# deps: numpy
# run: python 03_matrix_linear_algebra.py
# expected_runtime: <5s
# expected_output: 矩阵加/乘/inv/det/特征值/SVD/batch 矩阵乘结果
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.1.3 矩阵运算与线性代数)
# Interview hooks:
#   1. A * B 与 A @ B 在 NumPy 中有何本质区别？
#   2. SVD 分解 U/S/Vt 的形状如何由输入矩阵 M×N 决定？
#   3. np.linalg.solve 与矩阵求逆相比有何优势？

import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# ========== 基本运算 ==========
print(A + B)  # 逐元素加法
print(A * B)  # 逐元素乘法（Hadamard 积）
print(A @ B)  # 矩阵乘法
print(A.dot(B))  # 等价于 A @ B

# ========== 矩阵属性 ==========
print(f"转置: \n{A.T}")
print(f"逆矩阵: \n{np.linalg.inv(A)}")
print(f"行列式: {np.linalg.det(A)}")
print(f"迹: {np.trace(A)}")
print(f"秩: {np.linalg.matrix_rank(A)}")
print(f"特征值: {np.linalg.eigvals(A)}")

# ========== 常用矩阵操作 ==========
# 解线性方程组 Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # x = [2, 3]
print(f"解: {x}")

# SVD 分解（降维、推荐系统核心）
np.random.seed(42)
A = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
# 低秩近似
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# 广播在矩阵运算中的应用：批量矩阵乘法
batch_A = np.random.randn(10, 3, 3)  # 10 个 3x3 矩阵
batch_x = np.random.randn(10, 3, 1)  # 10 个向量
# 广播后逐元素矩阵乘法
batch_result = batch_A @ batch_x  # (10, 3, 1)

if __name__ == "__main__":
    print("OK")
