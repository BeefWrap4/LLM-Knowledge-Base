# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.5.2 ROC 曲线与 AUC
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn, matplotlib
# run: python 10_roc_auc_curve.py
# expected_runtime: <10s
# expected_output: confusion matrix, classification report, ROC-AUC value
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. ROC-AUC 的统计意义是什么?（正样本得分高于负样本的概率）
# 2. 类别不均衡时为什么 PR-AUC 比 ROC-AUC 更合适?
# 3. 如何根据业务场景选择分类阈值?

import os
import matplotlib
matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

def main():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # 逻辑回归 + 评估
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:\n", cm)

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # ROC-AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {auc:.4f}")

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6), facecolor='white')
    plt.plot(fpr, tpr, color='#4A6FA5', lw=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#7A8B99', lw=1, linestyle='--', label='Random')
    plt.fill_between(fpr, tpr, alpha=0.15, color='#4A6FA5')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, color='#333333')
    plt.legend(loc='lower right')
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存: {out_path}")


if __name__ == "__main__":
    main()
