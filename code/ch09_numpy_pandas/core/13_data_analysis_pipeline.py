# ---
# chapter: 9
# topic: NumPy 与 Pandas 数据处理
# topic_id: numpy_pandas.data_analysis_pipeline
# difficulty: 高
# tier: core
# deps: pandas, numpy, matplotlib, seaborn
# run: python 13_data_analysis_pipeline.py
# expected_runtime: <20s
# expected_output: 模拟电商数据集 6 步分析；仅 --output 时保存 PNG
# ---
# See: ../../../09_NumPy与Pandas数据处理.md
# Interview hooks:
#   1. 描述一份完整的数据分析 pipeline 应该包含哪些步骤？
#   2. pd.cut 与 pd.qcut 的核心区别？分位数离散化适合什么场景？
#   3. 当数据存在 99 分位数之外的离群点时，为什么更推荐 clip 而非直接删除？

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ========== 步骤 1: 加载数据 ==========
# 生成模拟电商数据
np.random.seed(42)
n = 10000

df = pd.DataFrame(
    {
        "user_id": range(1, n + 1),
        "age": np.random.randint(18, 65, n),
        "gender": np.random.choice(["M", "F", "Unknown"], n, p=[0.48, 0.48, 0.04]),
        "city": np.random.choice(["BJ", "SH", "GZ", "SZ", "HZ", "CD"], n),
        "purchase_amount": np.random.exponential(500, n),
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Books"], n),
        "is_member": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="min"),
    }
)

# 添加一些缺失值和异常值
df.loc[np.random.choice(n, 100, replace=False), "age"] = np.nan
df.loc[np.random.choice(n, 50, replace=False), "purchase_amount"] *= 10

print(f"数据集大小: {df.shape}")
print(df.head())

# ========== 步骤 2: 数据探索 ==========
print("\n=== 数据类型 ===")
print(df.dtypes)

print("\n=== 描述统计 ===")
print(df.describe())

print("\n=== 缺失值 ===")
print(df.isnull().sum())

# ========== 步骤 3: 数据清洗 ==========
# 处理缺失值
df["age"] = df["age"].fillna(df["age"].median())

# 处理异常值（截断到 99% 分位数）
amount_99 = df["purchase_amount"].quantile(0.99)
df["purchase_amount"] = df["purchase_amount"].clip(upper=amount_99)

# ========== 步骤 4: 特征工程 ==========
# 年龄分段
df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 100], labels=["18-25", "26-35", "36-45", "46+"])

# 消费等级
df["spending_level"] = pd.qcut(df["purchase_amount"], q=4, labels=["低", "中", "高", "很高"])

# ========== 步骤 5: 分析 & 可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 各城市消费金额
city_sales = df.groupby("city")["purchase_amount"].sum().sort_values(ascending=False)
city_sales.plot(kind="bar", ax=axes[0, 0], color="#4A6FA5")
axes[0, 0].set_title("Purchase Amount by City")
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. 年龄组分布
df["age_group"].value_counts().plot(kind="pie", ax=axes[0, 1], autopct="%1.1f%%")
axes[0, 1].set_title("Age Group Distribution")

# 3. 会员 vs 非会员消费
df.boxplot(column="purchase_amount", by="is_member", ax=axes[1, 0])
axes[1, 0].set_title("Member vs Non-member Purchase")
plt.suptitle("")  # 去掉 boxplot 自动生成的 suptitle

# 4. 各类别消费趋势
category_daily = df.groupby([df["order_date"].dt.date, "category"])["purchase_amount"].sum().unstack()
category_daily.plot(ax=axes[1, 1])
axes[1, 1].set_title("Purchase Trend by Category")
axes[1, 1].legend(loc="upper left", fontsize=8)

plt.tight_layout()

# ========== 步骤 6: 输出洞察 ==========
print("\n=== 数据洞察 ===")
print(f"总用户数: {df['user_id'].nunique()}")
print(f"总消费金额: ¥{df['purchase_amount'].sum():,.0f}")
print(f"人均消费: ¥{df['purchase_amount'].mean():.0f}")
print(f"会员比例: {df['is_member'].mean() * 100:.1f}%")
print(f"最热门品类: {df['category'].mode()[0]}")


def main(output_path: Path | None = None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"分析图已保存: {output_path}")
    else:
        print("分析图已在内存中构建；默认不落盘（使用 --output PATH 可显式保存）。")
    plt.close(fig)
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="电商数据分析完整流程")
    parser.add_argument("--output", type=Path, help="可选图片输出路径；默认不写文件")
    args = parser.parse_args()
    main(args.output)
else:
    plt.close(fig)
