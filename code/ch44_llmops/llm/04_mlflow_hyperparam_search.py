# ---
# chapter: 44
# topic: LLMOps 生命周期与持续交付
# topic_id: llmops.mlflow_hyperparam_search
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: mlflow
# run: python 04_mlflow_hyperparam_search.py
# expected_runtime: < 2s
# expected_output: Multiple nested runs logged; prints best run summary (mocked)
# ---
# See: ../../../44_LLMOps生命周期与持续交付.md
# Interview hooks:
#  - LLM 应用中哪些超参数对结果影响最大？
#  - 网格搜索 vs 贝叶斯优化在 LLM 场景下哪个更合适？
#  - 如何用 MLflow Nested Run 表达"父实验 + 子样本"的层级关系？

import itertools
import os

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore


def main():
    models = [
        item.strip()
        for item in os.environ.get("LLM_MODEL_CANDIDATES", "gpt-5.6-terra,gpt-5.6-sol").split(",")
        if item.strip()
    ]
    search_space = {
        "temperature": [0.0, 0.3, 0.7, 1.0],
        "model": models,
        "prompt_style": ["concise", "detailed", "chain_of_thought"],
    }

    # 默认只展示计划，不触发任何外部 Tracking 服务。
    if os.environ.get("LLM_MOCK") != "0" or os.environ.get("LLM_REAL_API") != "1":
        combinations = list(
            itertools.product(
                search_space["temperature"],
                search_space["model"],
                search_space["prompt_style"],
            )
        )
        print(f"[offline] planned_runs={len(combinations)}, models={models}")
        print("OK")
        return

    if mlflow is None:
        raise RuntimeError("LLM_REAL_API=1 requires mlflow")

    mlflow.set_experiment("hyperparam-search")

    # 网格搜索
    for temp, model, style in itertools.product(
        search_space["temperature"], search_space["model"], search_space["prompt_style"]
    ):
        with mlflow.start_run(run_name=f"{model}_t{temp}_{style}"):
            mlflow.log_params(
                {
                    "temperature": temp,
                    "model": model,
                    "prompt_style": style,
                }
            )

            # 模拟：在每个父 run 中对每个测试用例开一个 nested run
            test_cases = [
                {"id": "tc_001", "query": "Q1"},
                {"id": "tc_002", "query": "Q2"},
            ]
            for test_case in test_cases:
                with mlflow.start_run(run_name=test_case["id"], nested=True):
                    mlflow.log_param("test_case_id", test_case["id"])
                    mlflow.log_metric("score", 0.5)  # 模拟分数

    # 查找最佳实验（依赖 search_runs 真实执行结果）
    try:
        experiment = mlflow.get_experiment_by_name("hyperparam-search")
        if experiment is not None:
            best_run = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"],
                max_results=1,
            )
            if not best_run.empty:
                print(f"最佳实验: {best_run.iloc[0]['run_id']}")
                if "metrics.accuracy" in best_run.columns:
                    print(f"最佳准确率: {best_run.iloc[0]['metrics.accuracy']}")
    except Exception as e:
        print(f"查找最佳实验失败: {e}")
    print("OK")


if __name__ == "__main__":
    main()
