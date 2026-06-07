# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.2.1 线性回归 (Linear Regression)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: numpy, scikit-learn
# run: python 01_linear_regression.py
# expected_runtime: <5s
# expected_output: R^2 score, MSE, learned parameters, Lasso sparsity
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. 线性回归的解析解 w* = (X^T X)^{-1} X^T y 在什么情况下不可用?
# 2. L1 (Lasso) 与 L2 (Ridge) 正则化的几何解释与适用场景差异?
# 3. 为什么 Lasso 能产生稀疏解而 Ridge 只能将参数收缩到接近 0?

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 生成模拟数据
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([2.0, -1.5, 0.5, 3.0, -2.0])
    y = X @ true_w + np.random.randn(n_samples) * 0.5  # 添加噪声

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 普通线性回归
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"学习到的参数: {lr.coef_}")

    # L2 正则化（Ridge）— 防止过拟合
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # L1 正则化（Lasso）— 自动特征选择
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    print(f"Lasso 稀疏参数: {lasso.coef_}")  # 部分参数被压缩至0


if __name__ == "__main__":
    main()
    print("OK")
