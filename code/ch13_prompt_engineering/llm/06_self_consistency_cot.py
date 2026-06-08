# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.2.3 Self-Consistency
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai (via shared.llm_client)
# run: python 06_self_consistency_cot.py
# expected_runtime: 5-15s (real API)
# expected_output: 打印多次采样后的多数投票答案与置信度
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.2.3
# Interview hooks:
# - Self-consistency 为何要求 temperature > 0？
# - 采样次数 n 与准确率的关系？(收益递减)
# - 如何把 self-consistency 与 ToT 结合？

import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent  # /app/code or code/
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

import re
from collections import Counter

from shared.llm_client import UnifiedClient

_client = UnifiedClient()


def extract_final_answer(text: str) -> str:
    """从输出中提取最终答案。"""
    m = re.search(r"答案[:：]\s*(\S+)", text)
    if m:
        return m.group(1).rstrip("。.,，")
    return text.strip().split("\n")[-1]


def self_consistency_cot(prompt: str, n_samples: int = 5, temperature: float = 0.7):
    """
    Self-Consistency CoT：多次采样，多数投票

    Args:
        prompt: CoT prompt
        n_samples: 采样次数（建议 5-10 次）
        temperature: 必须 > 0 才能产生多样化推理路径
    """
    answers = []

    for _ in range(n_samples):
        resp = _client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,  # >0 以生成不同推理路径
        )
        content = resp.content
        answer = extract_final_answer(content)
        answers.append(answer)

    # 多数投票
    most_common = Counter(answers).most_common(1)[0]
    return most_common[0], most_common[1] / n_samples  # (答案, 置信度)


if __name__ == "__main__":
    prompt = "若 x + y = 50 且 x - y = 34，求 x 和 y。请逐步思考，最后给出 答案：x,y"
    answer, confidence = self_consistency_cot(prompt, n_samples=5, temperature=0.8)
    print(f"[多数投票答案] {answer}")
    print(f"[置信度] {confidence:.2%}")
