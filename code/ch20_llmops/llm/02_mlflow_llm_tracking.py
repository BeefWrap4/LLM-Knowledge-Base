# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.2.2 MLflow 核心概念
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: mlflow, openai (live requires LLM_MOCK=0 and LLM_REAL_API=1)
# run: python 02_mlflow_llm_tracking.py
# expected_runtime: < 1s (mocked) / depends on API (live)
# expected_output: Three MLflow runs logged with metrics; print accuracy/latency summary
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2022-mlflow-核心概念-⭐⭐⭐
# Interview hooks:
#  - MLflow 的 Experiment / Run / Parameter / Metric / Artifact 分别是什么？
#  - 为什么 Prompt 模板要作为 Artifact 保存？
#  - 如何用 MLflow 追踪多 Prompt 变体的对比实验？

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def _offline_client():
    """返回无网络的确定性客户端。"""

    class _Choice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class _Usage:
        total_tokens = 80

    class _Resp:
        def __init__(self, content, tokens=80):
            self.choices = [_Choice(content)]
            self.usage = _Usage()
            self.usage.total_tokens = tokens

    class _Mock:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kwargs):
                    del model, kwargs
                    user_msg = messages[-1]["content"].lower()
                    if "love" in user_msg:
                        ans = "positive"
                    elif "worst" in user_msg:
                        ans = "negative"
                    else:
                        ans = "neutral"
                    return _Resp(f"Sentiment: {ans}", tokens=60)

    return _Mock()


def main():
    live_api = os.environ.get("LLM_REAL_API") == "1" and os.environ.get("LLM_MOCK") == "0"
    if mlflow is None:
        print("mlflow not installed — install via `pip install mlflow` to run for real")
        print("OK")
        return

    # 1. 设置 MLflow Tracking URI
    # 离线模式强制使用内存 SQLite，不读取外部 Tracking 配置。
    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        if live_api
        else "sqlite:///:memory:"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("llm-sentiment-analysis")

    # 2. 定义实验参数
    prompt_variants = {
        "v1_basic": "Classify the sentiment of the following text as positive, negative, or neutral: {text}",
        "v2_cot": "Let's think step by step. First identify key emotional words, then classify the overall sentiment as positive, negative, or neutral. Text: {text}",
        "v3_expert": "You are a sentiment analysis expert. Analyze the following text and classify its sentiment as positive, negative, or neutral. Provide reasoning. Text: {text}",
    }

    # 3. 默认离线；只有显式 opt-in 才构造真实客户端并由 SDK 读取凭据。
    if live_api:
        if OpenAI is None:
            raise RuntimeError("LLM_REAL_API=1 requires the openai package")
        client = OpenAI()
    else:
        client = _offline_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6")

    for prompt_name, prompt_template in prompt_variants.items():
        with mlflow.start_run(run_name=prompt_name):
            # 记录参数
            mlflow.log_params(
                {
                    "prompt_name": prompt_name,
                    "prompt_template": prompt_template,
                    "model": model,
                    "temperature": 0.1,
                    "max_tokens": 100,
                }
            )

            # 记录开始时间
            start_time = time.time()

            # 模拟 LLM 调用和评估
            test_cases = [
                ("I absolutely love this product!", "positive"),
                ("This is the worst experience ever.", "negative"),
                ("The meeting is scheduled for 3pm.", "neutral"),
            ]

            correct = 0
            total_tokens = 0
            total_latency = 0.0

            for text, expected in test_cases:
                prompt = prompt_template.format(text=text)
                t0 = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    **(
                        {"reasoning_effort": "none", "max_completion_tokens": 100}
                        if model.startswith("gpt-5.6")
                        else {"max_tokens": 100}
                    ),
                )
                latency = time.time() - t0

                result = response.choices[0].message.content.strip().lower()
                is_correct = expected in result

                if is_correct:
                    correct += 1
                total_tokens += response.usage.total_tokens
                total_latency += latency

            # 记录指标
            accuracy = correct / len(test_cases)
            avg_latency = total_latency / len(test_cases)
            avg_tokens = total_tokens / len(test_cases)

            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "avg_latency_ms": avg_latency * 1000,
                    "avg_tokens_per_call": avg_tokens,
                    "total_tokens": total_tokens,
                    "total_time_sec": time.time() - start_time,
                }
            )

            # 保存 Prompt 模板为 Artifact
            with TemporaryDirectory(prefix="ch20-mlflow-") as temp_dir:
                artifact = Path(temp_dir) / "current_prompt.txt"
                artifact.write_text(prompt_template, encoding="utf-8")
                mlflow.log_artifact(str(artifact), artifact_path="prompts")

            print(f"[{prompt_name}] Accuracy: {accuracy:.2%}, Latency: {avg_latency * 1000:.0f}ms")

    print("\n✅ 所有实验完成！运行 `mlflow ui` 查看结果")
    print("OK")


if __name__ == "__main__":
    main()
