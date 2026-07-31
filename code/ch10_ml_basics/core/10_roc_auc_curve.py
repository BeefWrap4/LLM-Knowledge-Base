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

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def main(output_path: Path | None = None):
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 逻辑回归 + 评估
    # 二分类使用 liblinear，避免依赖 SciPy L-BFGS-B 的弃用参数链路。
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
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
    plt.figure(figsize=(6, 6), facecolor="white")
    plt.plot(fpr, tpr, color="#4A6FA5", lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#7A8B99", lw=1, linestyle="--", label="Random")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#4A6FA5")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, color="#333333")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"ROC 曲线已保存: {output_path}")
    else:
        print("ROC 曲线已在内存中生成；默认不落盘（使用 --output PATH 可显式保存）。")
    plt.close()
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练分类器并绘制 ROC 曲线")
    parser.add_argument("--output", type=Path, help="可选图片输出路径；默认不写文件")
    args = parser.parse_args()
    main(args.output)
