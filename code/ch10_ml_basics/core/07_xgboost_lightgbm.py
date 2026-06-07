# ---
# chapter: 10
# topic: 机器学习基础
# section: 10.3.2 XGBoost 与 LightGBM
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: xgboost, lightgbm, scikit-learn
# run: python 07_xgboost_lightgbm.py
# expected_runtime: <30s
# expected_output: XGBoost and LightGBM test accuracies
# ---
# See: ../tutorial/10_机器学习基础.md
#
# Interview hooks:
# 1. XGBoost 与 LightGBM 在分裂算法上的核心差异（Level-wise vs Leaf-wise）?
# 2. LightGBM 的直方图算法相对 XGBoost 预排序的优势?
# 3. XGBoost 在 GBDT 基础上做了哪些关键改进?（二阶泰勒展开 + 正则化项）

import sys

# 依赖检测 — core tier 不强制安装 xgboost/lightgbm
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

if not (HAS_XGB and HAS_LGB):
    print(f"[SKIP] xgboost={HAS_XGB}, lightgbm={HAS_LGB} (需 llm+ tier 或 pip install xgboost lightgbm)")
    sys.exit(0)

def main():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,        # 行采样比例
        colsample_bytree=0.8, # 列采样比例
        reg_alpha=0.1,        # L1 正则化
        reg_lambda=1.0,       # L2 正则化
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    print(f"XGBoost: {xgb_model.score(X_test, y_test):.4f}")

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        num_leaves=31,        # LightGBM 特有参数
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    print(f"LightGBM: {lgb_model.score(X_test, y_test):.4f}")

    print("OK")

if __name__ == "__main__":
    main()
