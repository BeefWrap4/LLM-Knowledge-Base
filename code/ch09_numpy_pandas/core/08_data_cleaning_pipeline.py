# ---
# chapter: 9
# topic: NumPy 与 Pandas 数据处理
# topic_id: numpy_pandas.data_cleaning_pipeline
# difficulty: 高
# tier: core
# deps: pandas, numpy, scikit-learn
# run: python 08_data_cleaning_pipeline.py
# expected_runtime: <10s
# expected_output: 只在训练集拟合清洗/编码/标准化 Pipeline，并转换训练集与测试集
# ---
# See: ../../../09_NumPy与Pandas数据处理.md
# Interview hooks:
#   1. 缺失值占比超过 50% 的列该如何处理？
#   2. 为什么中位数、IQR 边界和标准化参数只能在训练集拟合？
#   3. 名义输入特征为什么不应直接使用 LabelEncoder？

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """构造可复用、避免泄漏的特征预处理器。"""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ]
    )


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
   - One-Hot：名义类别，测试集未知类别用 handle_unknown="ignore"
   - OrdinalEncoder：仅用于明确有序类别，并显式定义顺序
   - Target：必须在交叉验证折内拟合，避免目标泄漏
   - LabelEncoder 通常用于目标 y，而不是名义输入特征
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
    # 去重可在划分前完成；任何会学习统计量的步骤都在划分后 fit。
    demo_df = demo_df.drop_duplicates().copy()
    train_df, test_df = train_test_split(demo_df, test_size=0.25, random_state=42)
    preprocessor = build_preprocessor(
        numeric_features=["age", "salary"],
        categorical_features=["dept", "city"],
    )
    train_ready = preprocessor.fit_transform(train_df)
    test_ready = preprocessor.transform(test_df)
    assert train_ready.shape[1] == test_ready.shape[1]
    print(f"train shape: {train_ready.shape}; test shape: {test_ready.shape}")
    print("OK")
