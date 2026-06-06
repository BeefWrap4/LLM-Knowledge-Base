# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.2.4 随机森林 (Random Forest)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 04_random_forest.py
# expected_runtime: <10s
# expected_output: classification report (precision/recall/f1 for each class)
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. 随机森林为什么比单棵决策树效果好?（降低方差）
# 2. Bagging 中 "数据随机" 与 "特征随机" 分别通过什么实现?
# 3. OOB (Out-of-Bag) 误差是什么? 为什么能替代交叉验证?

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100,    # 树的数量
        max_depth=5,         # 限制单棵树深度
        max_features='sqrt', # 每次分裂随机选 sqrt(m) 个特征
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("OK")

if __name__ == "__main__":
    main()
