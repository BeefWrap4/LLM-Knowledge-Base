# ---
# chapter: 9
# topic: NumPy 与 Pandas 数据处理
# topic_id: numpy_pandas.series_dataframe
# difficulty: 中
# tier: core
# deps: pandas, numpy
# run: python 04_series_dataframe.py
# expected_runtime: <5s
# expected_output: Series 与 DataFrame 的基本访问与 dtype 输出
# ---
# See: ../../../09_NumPy与Pandas数据处理.md
# Interview hooks:
#   1. Series 与 DataFrame 的区别是什么？DataFrame 内部如何存储每一列？
#   2. 什么是 Index？它在 DataFrame 中起什么作用？
#   3. 从 NumPy 数组创建带时间索引的 DataFrame 怎么做？

import numpy as np
import pandas as pd

# ========== Series ==========
s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"], name="numbers")
print(s["a"])  # 1
print(s.values)  # [1 2 3 4]
print(s.index)  # Index(['a', 'b', 'c', 'd'], dtype='object')

# ========== DataFrame 创建 ==========
df = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Carol", "David"],
        "age": [25, 30, 28, 35],
        "city": ["BJ", "SH", "GZ", "SZ"],
        "salary": [15000.0, 20000.0, 18000.0, 25000.0],
    }
)

# 从 NumPy 数组创建
df2 = pd.DataFrame(
    np.random.randn(5, 3), columns=["A", "B", "C"], index=pd.date_range("2024-01-01", periods=5)
)

if __name__ == "__main__":
    print("OK")
