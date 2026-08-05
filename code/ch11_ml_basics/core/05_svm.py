# ---
# chapter: 11
# topic: 机器学习基础
# topic_id: ml_basics.svm
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 05_svm.py
# expected_runtime: <5s
# expected_output: linear-kernel and RBF-kernel SVM accuracy scores
# ---
# See: ../../../11_机器学习基础.md
#
# Interview hooks:
# 1. SVM 的核函数是什么? 为什么需要核函数?（隐式高维映射）
# 2. 软间隔 SVM 中参数 C 的作用是什么?（容错 vs 间隔）
# 3. RBF 核中的 gamma 参数对模型复杂度有何影响?

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # 线性核 SVM
    svm_linear = SVC(kernel="linear", C=1.0)
    svm_linear.fit(X_train, y_train)

    # RBF 核 SVM — 适合非线性问题
    svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm_rbf.fit(X_train, y_train)

    print(f"线性核 SVM: {svm_linear.score(X_test, y_test):.4f}")
    print(f"RBF 核 SVM: {svm_rbf.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
    print("OK")
