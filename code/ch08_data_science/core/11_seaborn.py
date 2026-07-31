# ---
# chapter: 8
# topic: 数据科学核心库 - Seaborn 统计可视化
# section: 8.3.2
# difficulty: 中
# tier: core
# deps: seaborn, matplotlib, pandas, numpy
# run: python 11_seaborn.py
# expected_runtime: <5s
# expected_output: 箱线图/热力图/分布图/分类散点图在内存中构建；仅 --output 时保存
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.3.2 Seaborn 统计可视化)
# Interview hooks:
#   1. 箱线图（Box Plot）展示的 5 个统计量是什么？1.5*IQR 的作用？
#   2. heatmap 中 annot / cmap / center 三个参数各代表什么？
#   3. histplot(kde=True) 与 displot(kind='kde') 之间的差异？

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 生成示例数据
np.random.seed(42)
df = pd.DataFrame(
    {
        "x": np.random.randn(200),
        "y": np.random.randn(200),
        "category": np.random.choice(["A", "B", "C"], 200),
        "value": np.random.exponential(2, 200),
    }
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 箱线图（Box Plot）— 面试常考：箱线图的含义
# 避免 melt 默认 value_name='value' 与原列名冲突
df_melted = pd.melt(df, id_vars=["category"], value_vars=["value"], value_name="val")
sns.boxplot(
    data=df_melted,
    x="category",
    y="val",
    hue="category",
    legend=False,
    ax=axes[0, 0],
    palette="Blues",
)
axes[0, 0].set_title("Box Plot: Median, Quartiles, Outliers")

# 2. 热力图（Correlation Heatmap）
np.random.seed(42)
corr_df = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", center=0, ax=axes[0, 1])
axes[0, 1].set_title("Correlation Heatmap")

# 3. 分布图（Distribution Plot）
sns.histplot(df["x"], kde=True, ax=axes[1, 0], color="#4A6FA5")
axes[1, 0].set_title("Distribution with KDE")

# 4. 散点图 + 回归线
sns.scatterplot(data=df, x="x", y="y", hue="category", ax=axes[1, 1])
axes[1, 1].set_title("Scatter Plot by Category")

plt.tight_layout()


def main(output_path: Path | None = None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {output_path}")
    else:
        print("图表已在内存中构建；默认不落盘（使用 --output PATH 可显式保存）。")
    plt.close(fig)
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seaborn 统计可视化示例")
    parser.add_argument("--output", type=Path, help="可选图片输出路径；默认不写文件")
    args = parser.parse_args()
    main(args.output)
else:
    plt.close(fig)
