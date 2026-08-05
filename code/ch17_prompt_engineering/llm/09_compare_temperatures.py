# ---
# chapter: 17
# topic: Prompt Engineering
# topic_id: prompt_engineering.compare_temperatures
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai (via shared.llm_client)
# run: python 09_compare_temperatures.py
# expected_runtime: 5-20s (real api)
# expected_output: 打印不同 temperature 下的 3 个样本对比
# ---
# See: ../../../17_Prompt_Engineering.md
# Interview hooks:
# - Temperature 的数学原理是什么？(logits / T)
# - T=0 是否完全等价于 greedy decoding？为什么？
# - 高 Temperature 下的"重复惩罚"在哪类任务尤其重要？

import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent  # /app/code or code/
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

from shared.llm_client import UnifiedClient

_client = UnifiedClient()


def compare_temperatures(prompt: str, temps=None):
    """对比不同 Temperature 下的输出差异"""
    if temps is None:
        temps = [0.0, 0.5, 1.0]
    results = {}
    for t in temps:
        results[t] = [
            _client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=t,
            ).content
            for _ in range(3)  # 每个温度生成3个样本
        ]
    return results


if __name__ == "__main__":
    # 示例：Temperature=0 时三次输出完全相同；Temperature=1 时三次输出各不相同
    results = compare_temperatures("用一句话形容秋天", [0.0, 0.7, 1.2])
    for t, samples in results.items():
        print(f"\n--- Temperature={t} ---")
        for i, s in enumerate(samples, 1):
            print(f"  样本{i}: {s}")
    print("\nOK")
