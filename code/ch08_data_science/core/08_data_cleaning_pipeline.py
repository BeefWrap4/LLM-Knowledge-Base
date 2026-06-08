# ---
# chapter: 8
# topic: 数据科学核心库 - 完整数据清洗流程
# section: 8.2.4 (面试高频题)
# difficulty: 高
# tier: core
# deps: pandas, numpy, scikit-learn
# run: python 08_data_cleaning_pipeline.py
# expected_runtime: <10s
# expected_output: 缺失值统计、IQR 截断、删除重复值、类别编码与标准化后的 DataFrame
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.2.4 数据清洗完整流程)
# Interview hooks:
#   1. 缺失值占比超过 50% 的列该如何处理？
#   2. IQR 异常值处理有哪几种策略（删除 / clip / 变换 / 鲁棒模型）？
#   3. 类别特征 One-Hot 与 Label 编码的适用场景分别是什么？

import numpy as np
import pandas as pd


def data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    完整的数据清洗流程（面试常考）

    步骤：
    1. 处理缺失值
    2. 处理异常值
    3. 处理重复值
    4. 类型转换
    5. 特征编码
    6. 特征标准化
    """
    df = df.copy()

    # Step 1: 处理缺失值
    print(f"缺失值统计:\n{df.isnull().sum()}")

    # 数值列：用中位数填充（对异常值更鲁棒）
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # 类别列：用众数填充
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Step 2: 处理异常值（IQR 方法）
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # 可选：删除异常值或用边界值替换
        # df = df[(df[col] >= lower) & (df[col] <= upper)]
        df[col] = df[col].clip(lower, upper)  # 用边界值替换

    # Step 3: 处理重复值
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"删除重复值: {before} -> {len(df)}")

    # Step 4: 类型转换
    # 将可以转为数值的列转换
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Step 5: 类别特征编码
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() < 10:  # 低基数类别
            # One-Hot 编码
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        else:
            # Label 编码
            df[col] = le.fit_transform(df[col].astype(str))

    # Step 6: 数值特征标准化
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


# ========== 面试常考点：数据清洗中的决策 ==========
"""
Q: 缺失值很多（>50%）的列怎么处理？
A: 通常删除该列，除非该列非常重要。可用缺失值本身作为特征（is_missing 标志）。

Q: 异常值怎么处理？
A: 先检查是否为数据录入错误。如果不是：
   - 删除（损失数据）
   - 截断（clip 到边界值）
   - 对数变换（减小极端值影响）
   - 使用对异常值鲁棒的模型（如树模型）

Q: 类别特征怎么做编码？
A:
   - One-Hot：低基数（<10），无序类别
   - Label：高基数，有序类别
   - Target：高基数，与目标变量相关
"""


if __name__ == "__main__":
    # 构造一个含缺失/重复/类别/数值的混合 DataFrame 用于演示
    demo_df = pd.DataFrame(
        {
            "age": [25, np.nan, 35, 25, 200, 40, 35, np.nan, 28],
            "salary": [5000.0, 6000.0, 7000.0, 5000.0, 999999.0, 8000.0, 7000.0, 6500.0, 5500.0],
            "dept": ["Tech", "HR", "Tech", "Tech", "Sales", "HR", "Tech", "Sales", None],
            "city": ["BJ", "SH", "BJ", "BJ", "GZ", "SH", "BJ", "GZ", "SH"],
        }
    )
    cleaned = data_cleaning_pipeline(demo_df)
    print(cleaned.head())
