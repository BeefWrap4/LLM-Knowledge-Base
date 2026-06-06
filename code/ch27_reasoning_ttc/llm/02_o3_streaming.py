# ---
# chapter: 27
# topic: o3 streaming with hidden reasoning tokens
# section: 27.2 Reasoning Effort API
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=1.40.0 (optional)
# run: python 02_o3_streaming.py
# expected_runtime: <2s
# expected_output: streams a mock CoT, prints delta events
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.2
# Interview hooks:
#   1. o3 思维链默认是否对用户可见？用什么参数控制？
#   2. 流式场景下如何区分 thought delta 和 answer delta？
#   3. 长 CoT 流式对 UX 的影响（首字延迟 vs 总延迟）？
"""o3 流式输出：分阶段打印 reasoning 与 final answer delta。

OpenAI o3 在流式中不会把 reasoning tokens 直接暴露给客户端——客户端只能
通过 `completion_tokens_details.reasoning_tokens` 在 final chunk 看到总
计数。这里 mock 一个简化的"分阶段"流，便于讲解协议。
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass


@dataclass
class StreamDelta:
    """单条流式 chunk。"""

    phase: str  # "reasoning" | "answer"
    text: str
    tokens: int


def _mock_stream(question: str, thought_chars: int = 400) -> list[StreamDelta]:
    """生成模拟流：先吐 reasoning，再吐 answer。"""
    random.seed(len(question))
    thought = (
        "Let me think step by step. First, I need to consider the constraints... "
        * (thought_chars // 60)
    )
    thought = thought[:thought_chars]
    answer = "Therefore, the answer is 42."

    # 切成 ~20 token 的小块
    out: list[StreamDelta] = []
    for i in range(0, len(thought), 60):
        out.append(StreamDelta("reasoning", thought[i : i + 60], 15))
    for i in range(0, len(answer), 10):
        out.append(StreamDelta("answer", answer[i : i + 10], 3))
    return out


def stream_o3(question: str) -> None:
    """真实环境调用 openai 流式；缺依赖则 mock。"""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            stream = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": question}],
                reasoning_effort="high",
                max_completion_tokens=50_000,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
            return
        except Exception as e:  # noqa: BLE001
            print(f"[warn] streaming failed: {e}, fallback to mock")

    print(f"[Q] {question}\n")
    t0 = time.perf_counter()
    n_reason, n_ans = 0, 0
    for d in _mock_stream(question):
        # 思维链用灰色标记(终端不支持就明文展示)
        if d.phase == "reasoning":
            n_reason += d.tokens
        else:
            n_ans += d.tokens
        print(d.text, end="", flush=True)
        time.sleep(0.005)
    print(f"\n\n[stats] reasoning_tokens={n_reason}  answer_tokens={n_ans}  "
          f"elapsed={time.perf_counter() - t0:.2f}s")


def main() -> None:
    stream_o3("Prove that sqrt(2) is irrational.")
    print("OK")


if __name__ == "__main__":
    main()
