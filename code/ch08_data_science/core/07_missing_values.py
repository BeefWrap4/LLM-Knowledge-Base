# ---
# chapter: 8
# topic: 数据科学核心库 - Pandas 缺失值处理
# section: 8.2.4
# difficulty: 高
# tier: core
# deps: pandas, numpy
# run: python 07_missing_values.py
# expected_runtime: <5s
# expected_output: 缺失值检测/删除/填充示例（fillna 链式调用即可观察）
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.2.4 缺失值处理)
# Interview hooks:
#   1. dropna 中 subset / how / thresh / axis 各自代表什么？
#   2. ffill / bfill / interpolate 三种填充方式各适用于什么场景？
#   3. 为什么数值列常用中位数而非均值填充？

import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', None, 'c', 'd', 'e'],
    'C': [10, 20, 30, np.nan, 50],
    'D': [1.0, 2.0, 3.0, 4.0, 5.0]
})

# ========== 检测缺失值 ==========
print(df.isnull())        # 逐元素判断是否为缺失值
print(df.isnull().sum())  # 每列缺失值数量
print(df.isnull().mean()) # 每列缺失值比例

# ========== 删除缺失值 ==========
df.dropna()                    # 删除包含 NaN 的行
df.dropna(subset=['A', 'C'])   # 只删除 A 或 C 为 NaN 的行
df.dropna(how='all')           # 只删除全为 NaN 的行
df.dropna(axis=1)              # 删除包含 NaN 的列
df.dropna(thresh=3)            # 保留至少 3 个非 NaN 值的行

# ========== 填充缺失值 ==========
df['A'].fillna(0)              # 用 0 填充
df['A'].fillna(df['A'].mean()) # 用均值填充
df['A'].fillna(method='ffill') # 前向填充（用前一个有效值）
df['A'].fillna(method='bfill') # 后向填充（用后一个有效值）
df['A'].interpolate()          # 线性插值

# ========== 不同列用不同策略 ==========
df.fillna({
    'A': df['A'].mean(),       # 数值列用均值
    'B': 'unknown',            # 类别列用默认值
    'C': df['C'].median()      # 用中位数
})

if __name__ == "__main__":
    print("OK")
