# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.4.1 K-Means 聚类
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn, numpy
# run: python 08_kmeans.py
# expected_runtime: <10s
# expected_output: best K value from silhouette score
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. K-Means 如何选择 K 值?（肘部法则 + 轮廓系数）
# 2. K-Means 对初始质心敏感, K-Means++ 如何改进初始化?
# 3. K-Means 的核心缺陷与对应的改进算法?（DBSCAN / K-Medoids）

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    # 生成聚类数据
    np.random.seed(42)
    X_cluster = np.vstack(
        [
            np.random.randn(100, 2) + [0, 0],
            np.random.randn(100, 2) + [5, 5],
            np.random.randn(100, 2) + [0, 5],
        ]
    )

    # 肘部法则选择 K
    inertias = []
    silhouettes = []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_cluster, km.labels_))

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"最优 K 值（轮廓系数）: {best_k}")

    # 最终聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)
    print(f"聚类标签分布: {np.bincount(labels)}")


if __name__ == "__main__":
    main()
    print("OK")
