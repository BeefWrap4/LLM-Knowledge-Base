# ---
# chapter: 8
# topic: 数据科学核心库 - Pandas groupby/apply/map
# section: 8.2.3
# difficulty: 高
# tier: core
# deps: pandas, numpy
# run: python 06_groupby_apply_map.py
# expected_runtime: <5s
# expected_output: 部门聚合结果、transform/filter/map/apply 输出
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.2.3 groupby、apply、map、applymap 区别)
# Interview hooks:
#   1. groupby + agg / transform / filter 三件套各自返回什么形状？
#   2. map 与 apply 的核心区别是什么？为什么说 apply 是 Python 级循环？
#   3. applymap 当前的状态是什么？替代方案是什么？

import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "department": ["Tech", "Tech", "HR", "HR", "Sales", "Sales"],
        "employee": ["Alice", "Bob", "Carol", "David", "Eve", "Frank"],
        "salary": [15000, 20000, 12000, 14000, 18000, 22000],
        "bonus": [3000, 5000, 2000, 2500, 4000, 6000],
    }
)

# ========== groupby：分组聚合 ==========
# 按部门分组，计算平均工资
dept_avg = df.groupby("department")["salary"].mean()
print(dept_avg)

# 多列聚合
dept_stats = df.groupby("department").agg({"salary": ["mean", "sum", "count"], "bonus": ["mean", "max"]})
print(dept_stats)

# transform：保持原 DataFrame 形状
df["dept_avg_salary"] = df.groupby("department")["salary"].transform("mean")

# filter：筛选满足条件的组
depts = df.groupby("department").filter(lambda x: x["salary"].sum() > 30000)

# ========== map：Series 的逐元素映射 ==========
# 对 Series 的每个元素应用映射
df["dept_code"] = df["department"].map({"Tech": "T", "HR": "H", "Sales": "S"})

# 使用函数映射
df["salary_level"] = df["salary"].map(lambda x: "High" if x > 18000 else "Low")

# ========== apply：DataFrame/Series 的灵活应用 ==========
# Series.apply: 对每行应用函数
print(df["salary"].apply(lambda x: x * 1.1))  # 加薪 10%

# DataFrame.apply: axis=0 对每列，axis=1 对每行
print(df[["salary", "bonus"]].apply(np.sum, axis=0))  # 列求和
print(df[["salary", "bonus"]].apply(np.sum, axis=1))  # 行求和


# 返回 Series 的 apply（更复杂操作）
def total_compensation(row):
    return row["salary"] + row["bonus"] + row.get("extra", 0)


df["total"] = df.apply(total_compensation, axis=1)

# ========== applymap：DataFrame 逐元素操作（已弃用，用 map） ==========
df[["salary", "bonus"]] = df[["salary", "bonus"]].map(lambda x: f"${x:,.0f}")

if __name__ == "__main__":
    print("OK")
