# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.2.2 MLflow 核心概念
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: mlflow, openai (mocked fallback if unavailable)
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

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore

try:
    from openai import OpenAI
    _HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _mock_client():
    """返回一个 mock client 用于离线运行。"""
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
                def create(model, messages, temperature=0.0, max_tokens=100):
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
    if mlflow is None:
        print("mlflow not installed — install via `pip install mlflow` to run for real")
        return

    # 1. 设置 MLflow Tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("llm-sentiment-analysis")

    # 2. 定义实验参数
    prompt_variants = {
        "v1_basic": "Classify the sentiment of the following text as positive, negative, or neutral: {text}",
        "v2_cot": "Let's think step by step. First identify key emotional words, then classify the overall sentiment as positive, negative, or neutral. Text: {text}",
        "v3_expert": "You are a sentiment analysis expert. Analyze the following text and classify its sentiment as positive, negative, or neutral. Provide reasoning. Text: {text}",
    }

    # 3. 执行实验
    client = OpenAI() if _HAS_OPENAI and OpenAI is not None else _mock_client()

    for prompt_name, prompt_template in prompt_variants.items():
        with mlflow.start_run(run_name=prompt_name):
            # 记录参数
            mlflow.log_params({
                "prompt_name": prompt_name,
                "prompt_template": prompt_template,
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 100,
            })

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
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100,
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

            mlflow.log_metrics({
                "accuracy": accuracy,
                "avg_latency_ms": avg_latency * 1000,
                "avg_tokens_per_call": avg_tokens,
                "total_tokens": total_tokens,
                "total_time_sec": time.time() - start_time,
            })

            # 保存 Prompt 模板为 Artifact
            with open("current_prompt.txt", "w") as f:
                f.write(prompt_template)
            mlflow.log_artifact("current_prompt.txt")

            print(f"[{prompt_name}] Accuracy: {accuracy:.2%}, Latency: {avg_latency*1000:.0f}ms")

    print("\n✅ 所有实验完成！运行 `mlflow ui` 查看结果")


if __name__ == "__main__":
    main()
    print("OK")
