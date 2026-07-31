# ---
# chapter: 8
# topic: 数据科学核心库 - Pandas 内置绘图
# section: 8.3.3
# difficulty: 中
# tier: core
# deps: pandas, numpy, matplotlib
# run: python 12_pandas_plotting.py
# expected_runtime: <5s
# expected_output: 线图/面积图/柱状图/饼图在内存中构建；仅 --output 时保存
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.3.3 Pandas 内置绘图)
# Interview hooks:
#   1. ts.plot() 默认绘制的是什么类型图？能否一行代码切换为面积图？
#   2. plot.pie 的 autopct 参数控制什么？
#   3. Pandas 内置绘图与 Matplotlib 之间的关系是什么（封装关系）？

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成时间序列数据
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100)
ts = pd.DataFrame(
    {
        "sales": np.cumsum(np.random.randn(100)) + 100,
        "profit": np.cumsum(np.random.randn(100) * 0.5) + 20,
    },
    index=dates,
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 快速绘图（线图）
ts.plot(ax=axes[0, 0], title="Sales & Profit Trend")
axes[0, 0].set_title("Sales & Profit Trend")

# 面积图
ts.plot.area(ax=axes[0, 1], alpha=0.5, title="Area Chart")
axes[0, 1].set_title("Area Chart")

# 柱状图
ts.head(10).plot.bar(ax=axes[1, 0], title="Bar Chart")
axes[1, 0].set_title("Bar Chart (first 10 days)")

# 饼图
ts.iloc[-1].plot.pie(ax=axes[1, 1], autopct="%1.1f%%", title="Composition")
axes[1, 1].set_title("Composition (last day)")
axes[1, 1].set_ylabel("")

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
    parser = argparse.ArgumentParser(description="Pandas 内置绘图示例")
    parser.add_argument("--output", type=Path, help="可选图片输出路径；默认不写文件")
    args = parser.parse_args()
    main(args.output)
else:
    plt.close(fig)
