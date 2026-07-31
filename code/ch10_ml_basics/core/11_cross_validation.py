# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.5.3 交叉验证 (Cross-Validation)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn
# run: python 11_cross_validation.py
# expected_runtime: <10s
# expected_output: per-fold AUC scores, mean ± std
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. K-Fold 交叉验证的优缺点? K 值如何选择?
# 2. StratifiedKFold 与普通 KFold 的区别? 分类任务为什么首选 StratifiedKFold?
# 3. 留一法 (Leave-One-Out) 的适用场景与计算代价?

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def main():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)

    # 5-Fold 分层交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=skf, scoring="roc_auc")

    print(f"5-Fold AUC: {cv_scores}")
    print(f"均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


if __name__ == "__main__":
    main()
    print("OK")
