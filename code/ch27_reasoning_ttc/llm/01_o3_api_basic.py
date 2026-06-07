# ---
# chapter: 27
# topic: OpenAI o3 API basic usage
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0 (optional, mock mode if absent)
# run: python 01_o3_api_basic.py
# expected_runtime: <2s
# expected_output: prints mock reasoning trace and final answer
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2
# Interview hooks:
#   1. o3 / o4-mini 的 reasoning_effort 参数含义？三档对应什么？
#   2. 为什么 o-series 不能用 temperature=0 的常见做法？默认采样超参是什么？
#   3. reasoning_effort="high" 与 max_completion_tokens 的关系？
"""OpenAI o3 API 基础调用：reasoning_effort 三档对比。

Mock 模式：当未设置 OPENAI_API_KEY 或 import openai 失败时，使用本地伪推理引擎
演示 reasoning_effort 对思维链长度与最终答案的影响。
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class ReasoningResponse:
    """统一封装 o3 / o4-mini 风格的返回结构。"""

    model: str
    effort: str
    thought_tokens: int
    answer: str
    total_tokens: int


# 模拟 o3 推理：effort 越高，thought 越长，答案准确率越高
_EFFORT_PROFILE = {
    "low": {"thought": (100, 500), "accuracy": 0.55, "latency_s": 1.5},
    "medium": {"thought": (1000, 5000), "accuracy": 0.80, "latency_s": 6.0},
    "high": {"thought": (10000, 50000), "accuracy": 0.95, "latency_s": 45.0},
}


def _mock_reason(question: str, effort: str, seed: int = 0) -> ReasoningResponse:
    """伪推理：随机生成"思考链"长度并按档位概率给出正确答案。"""
    random.seed(seed + hash(question) % 1000)
    profile = _EFFORT_PROFILE[effort]
    n_thought = random.randint(*profile["thought"])
    correct = random.random() < profile["accuracy"]
    answer = "42" if correct else "41"  # 经典示例问题
    return ReasoningResponse(
        model="o3-mini",
        effort=effort,
        thought_tokens=n_thought,
        answer=answer,
        total_tokens=n_thought + len(answer.split()),
    )


def call_o3(question: str, effort: str = "medium") -> ReasoningResponse:
    """调用 o3-mini；若未配置 API key 则走 mock。"""
    if effort not in _EFFORT_PROFILE:
        raise ValueError(f"effort 必须是 {list(_EFFORT_PROFILE)}")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # lazy import

            client = OpenAI()
            resp = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": question}],
                reasoning_effort=effort,
                max_completion_tokens=50_000,
            )
            return ReasoningResponse(
                model="o3-mini",
                effort=effort,
                thought_tokens=resp.usage.completion_tokens_details.reasoning_tokens,
                answer=resp.choices[0].message.content,
                total_tokens=resp.usage.total_tokens,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] openai 调用失败 ({e})，回退到 mock")

    return _mock_reason(question, effort)


def main() -> None:
    q = "What is the meaning of life, the universe, and everything?"
    for effort in ("low", "medium", "high"):
        r = call_o3(q, effort=effort)
        print(
            f"[{effort:>6}] thought={r.thought_tokens:>6} tok  "
            f"answer={r.answer!r}  total={r.total_tokens}"
        )

    # 展示官方推荐参数
    print("\n官方推荐:")
    print("  • low    →  简单 QA, 路由, 分类")
    print("  • medium →  一般推理, 短代码")
    print("  • high   →  数学, 竞赛编程, 复杂证明")


if __name__ == "__main__":
    main()
