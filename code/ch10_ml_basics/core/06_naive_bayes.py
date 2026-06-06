# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.2.6 朴素贝叶斯 (Naive Bayes)
# difficulty: ⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 06_naive_bayes.py
# expected_runtime: <5s
# expected_output: Naive Bayes classification accuracy
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. 朴素贝叶斯的 "朴素" 假设是什么? 实际中是否成立?
# 2. 高斯 / 多项式 / 伯努利朴素贝叶斯分别适用于什么数据?
# 3. 朴素贝叶斯为何对小规模数据和高维稀疏数据（文本）效果好?

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print(f"朴素贝叶斯准确率: {nb.score(X_test, y_test):.4f}")

    print("OK")

if __name__ == "__main__":
    main()
