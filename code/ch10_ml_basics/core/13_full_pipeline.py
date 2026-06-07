# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.6 Scikit-learn 完整建模流程
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: scikit-learn, pandas
# run: python 13_full_pipeline.py
# expected_runtime: <60s
# expected_output: best params, CV/test AUC, classification report, feature importance
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. 为什么使用 sklearn Pipeline?（防止数据泄漏 + 流程封装）
# 2. 类别不均衡时如何处理?（class_weight / SMOTE / 阈值调整）
# 3. 特征重要性有哪些评估方法?（基于不纯度 / 基于置换 / SHAP）

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification

def main():
    # 1. 数据
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, weights=[0.7, 0.3],
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

    # 2. 构建 Pipeline — 防止数据泄漏
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 3. 超参数搜索
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    # 4. 评估
    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    print(f"最优参数: {grid.best_params_}")
    print(f"验证集最优 AUC: {grid.best_score_:.4f}")
    print(f"测试集 AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")

    # 5. 特征重要性
    rf = best_model.named_steps['classifier']
    importances = pd.Series(rf.feature_importances_)
    print(f"\n特征重要性:\n{importances.sort_values(ascending=False)}")


if __name__ == "__main__":
    main()
