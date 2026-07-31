# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.2.3 决策树 (Decision Tree)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 03_decision_tree.py
# expected_runtime: <5s
# expected_output: decision tree accuracy, feature importance, tree rules
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. 信息增益（ID3）vs 信息增益率（C4.5）vs 基尼系数（CART）的差异?
# 2. 决策树有哪些剪枝策略? 预剪枝 vs 后剪枝?
# 3. 为什么决策树对特征尺度不敏感, 但容易过拟合?

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # 决策树
    # max_depth: 限制树深度，防止过拟合
    # min_samples_split: 节点分裂所需最小样本数
    dt = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)

    print(f"决策树准确率: {dt.score(X_test, y_test):.4f}")
    print(f"特征重要性: {dict(zip(iris.feature_names, dt.feature_importances_))}")

    # 导出决策规则
    tree_rules = export_text(dt, feature_names=iris.feature_names)
    print(tree_rules[:500])


if __name__ == "__main__":
    main()
    print("OK")
