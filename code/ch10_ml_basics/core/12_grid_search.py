# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.5.4 超参数调优 (Grid Search)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 12_grid_search.py
# expected_runtime: <30s
# expected_output: best hyperparameters, best CV AUC
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. Grid Search 与 Random Search 各自的适用场景?
# 2. 贝叶斯优化（Bayesian Optimization）的核心思想?
# 3. n_jobs=-1 的含义? 为什么 GridSearch 容易并行化?

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV

def main():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # 网格搜索
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # liblinear 支持 l1 和 l2
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"最优参数: {grid.best_params_}")
    print(f"最优 AUC: {grid.best_score_:.4f}")

    print("OK")

if __name__ == "__main__":
    main()
