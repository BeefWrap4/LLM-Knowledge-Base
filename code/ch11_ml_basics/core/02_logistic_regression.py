# ---
# chapter: 11
# topic: 机器学习基础
# topic_id: ml_basics.logistic_regression
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 02_logistic_regression.py
# expected_runtime: <5s
# expected_output: probability range, accuracy score
# ---
# See: ../../../11_机器学习基础.md
#
# Interview hooks:
# 1. 逻辑回归为什么用交叉熵损失而不是 MSE?（非凸性 + 梯度消失）
# 2. Sigmoid 函数的输出为何天然适合作为概率?
# 3. 正则化系数 C 与 lambda 的关系是什么? C=1/lambda, C 越小正则化越强

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    # 生成分类数据
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 逻辑回归
    clf = LogisticRegression(max_iter=1000, C=1.0)  # C=1/λ，越小正则化越强
    clf.fit(X_train, y_train)

    # 预测概率
    proba = clf.predict_proba(X_test)
    print(f"类别 1 的概率范围: [{proba[:, 1].min():.3f}, {proba[:, 1].max():.3f}]")
    print(f"准确率: {clf.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
    print("OK")
