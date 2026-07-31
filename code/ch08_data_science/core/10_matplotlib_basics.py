# ---
# chapter: 8
# topic: 数据科学核心库 - Matplotlib 基础
# section: 8.3.1
# difficulty: 中
# tier: core
# deps: matplotlib, numpy
# run: python 10_matplotlib_basics.py
# expected_runtime: <5s
# expected_output: 4 宫格图表在内存中构建；仅 --output 时保存
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.3.1 Matplotlib 基础)
# Interview hooks:
#   1. fig, axes = plt.subplots(2, 2) 中 axes 的索引顺序是什么？
#   2. tight_layout 与 subplots_adjust 的区别？
#   3. axvline 与 axhline 在标注统计量时的常用场景？

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 非交互式后端，方便无显示器环境运行
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字号
plt.rcParams["font.size"] = 12

# ========== 基本绘图 ==========
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 线图
axes[0, 0].plot(x, y1, label="sin(x)", color="#4A6FA5", linewidth=2)
axes[0, 0].plot(x, y2, label="cos(x)", color="#e74c3c", linewidth=2, linestyle="--")
axes[0, 0].set_title("Line Plot")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 散点图
np.random.seed(42)
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), c="#2ecc71", alpha=0.6, s=100)
axes[0, 1].set_title("Scatter Plot")

# 柱状图
categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]
axes[1, 0].bar(categories, values, color=["#4A6FA5", "#6B8CBB", "#8BA3C7", "#2E4A62", "#7A8B99"])
axes[1, 0].set_title("Bar Chart")

# 直方图
data = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data, bins=30, color="#4A6FA5", edgecolor="white", alpha=0.7)
axes[1, 1].set_title("Histogram")
axes[1, 1].axvline(data.mean(), color="red", linestyle="--", label=f"Mean={data.mean():.2f}")
axes[1, 1].legend()

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
    parser = argparse.ArgumentParser(description="Matplotlib 四宫格示例")
    parser.add_argument("--output", type=Path, help="可选图片输出路径；默认不写文件")
    args = parser.parse_args()
    main(args.output)
else:
    plt.close(fig)
