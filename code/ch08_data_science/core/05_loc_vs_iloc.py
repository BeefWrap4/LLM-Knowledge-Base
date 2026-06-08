# ---
# chapter: 8
# topic: 数据科学核心库 - Pandas loc vs iloc
# section: 8.2.2
# difficulty: 高
# tier: core
# deps: pandas
# run: python 05_loc_vs_iloc.py
# expected_runtime: <5s
# expected_output: 标签与位置两种索引方式的输出
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.2.2 loc vs iloc 的区别)
# Interview hooks:
#   1. df.loc[0:2] 与 df.iloc[0:2] 的区别？默认整数索引下分别取几行？
#   2. at/iat 与 loc/iloc 在性能上有何差异？
#   3. 何时应避免使用已废弃的 df.ix？

import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]}, index=[10, 20, 30, 40])  # 自定义索引

# ========== loc: 基于标签 ==========
print(df.loc[10])  # 取索引为10的行
print(df.loc[10:30])  # 取索引 10 到 30 的行（包含30）
print(df.loc[10, "A"])  # 标量访问 → 1
print(df.loc[:, "A"])  # 取列 'A'
print(df.loc[10:20, ["A", "B"]])  # 多行多列

# 布尔索引
print(df.loc[df["A"] > 2])  # 条件筛选

# ========== iloc: 基于整数位置 ==========
print(df.iloc[0])  # 取第0行（即索引为10的行）
print(df.iloc[0:2])  # 取第0、1行（不包含第2行）
print(df.iloc[0, 0])  # 标量访问 → 1
print(df.iloc[:, 0])  # 取第0列
print(df.iloc[0:2, 0:2])  # 切片

# ========== 常见陷阱 ==========
# df.loc[0]  # KeyError! 索引中没有 0
# df.iloc[10]  # IndexError! 只有4行，没有第10行

# ========== 混合使用 ==========
# 先 iloc 选行，再 loc 选列
subset = df.iloc[0:2]  # 先按位置选行
result = subset.loc[:, ["A"]]  # 再按标签选列

# 使用 ix（已废弃，了解即可）
# df.ix[0]  # 根据上下文判断是标签还是位置 — 不推荐使用

if __name__ == "__main__":
    print("OK")
