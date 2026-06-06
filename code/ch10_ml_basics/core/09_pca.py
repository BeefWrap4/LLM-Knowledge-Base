# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.4.2 PCA 主成分分析
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 09_pca.py
# expected_runtime: <5s
# expected_output: original vs reduced dimensions, explained variance
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. PCA 的核心数学步骤?（协方差矩阵 + 特征值分解）
# 2. 为什么 PCA 之前必须做标准化?（消除量纲影响）
# 3. 如何选择保留的主成分数量?（累计方差贡献率 ≥ 95%）

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def main():
    X, _ = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               random_state=42)

    # 标准化 + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)  # 保留 95% 方差
    X_pca = pca.fit_transform(X_scaled)

    print(f"原始维度: {X.shape[1]}")
    print(f"降维后维度: {X_pca.shape[1]}")
    print(f"各主成分方差贡献: {pca.explained_variance_ratio_[:5]}")
    print(f"累计方差贡献: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

    print("OK")

if __name__ == "__main__":
    main()
