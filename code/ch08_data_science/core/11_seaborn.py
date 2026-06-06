# ---
# chapter: 8
# topic: 数据科学核心库 - Seaborn 统计可视化
# section: 8.3.2
# difficulty: 中
# tier: core
# deps: seaborn, matplotlib, pandas, numpy
# run: python 11_seaborn.py
# expected_runtime: <5s
# expected_output: 箱线图/热力图/分布图/分类散点图渲染并保存
# ---
# See: ../tutorial/08_Python数据科学核心库.md (Section 8.3.2 Seaborn 统计可视化)
# Interview hooks:
#   1. 箱线图（Box Plot）展示的 5 个统计量是什么？1.5*IQR 的作用？
#   2. heatmap 中 annot / cmap / center 三个参数各代表什么？
#   3. histplot(kde=True) 与 displot(kind='kde') 之间的差异？

import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'category': np.random.choice(['A', 'B', 'C'], 200),
    'value': np.random.exponential(2, 200)
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 箱线图（Box Plot）— 面试常考：箱线图的含义
# 避免 melt 默认 value_name='value' 与原列名冲突
df_melted = pd.melt(df, id_vars=['category'], value_vars=['value'], value_name='val')
sns.boxplot(data=df_melted, x='category', y='val', ax=axes[0, 0], palette='Blues')
axes[0, 0].set_title('Box Plot: 中位数、四分位数、异常值')

# 2. 热力图（Correlation Heatmap）
np.random.seed(42)
corr_df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Correlation Heatmap')

# 3. 分布图（Distribution Plot）
sns.histplot(df['x'], kde=True, ax=axes[1, 0], color='#4A6FA5')
axes[1, 0].set_title('Distribution with KDE')

# 4. 散点图 + 回归线
sns.scatterplot(data=df, x='x', y='y', hue='category', ax=axes[1, 1])
axes[1, 1].set_title('Scatter Plot by Category')

plt.tight_layout()
plt.savefig('seaborn_demo.png', dpi=150, bbox_inches='tight')
plt.close(fig)

if __name__ == "__main__":
    print("OK")
