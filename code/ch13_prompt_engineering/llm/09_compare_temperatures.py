# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.3.1 Temperature 对比
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai (可选，缺失则使用 mock)
# run: python 09_compare_temperatures.py
# expected_runtime: <1s (mock) / 5-20s (real api)
# expected_output: 打印不同 temperature 下的 3 个样本对比
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.3.1
# Interview hooks:
# - Temperature 的数学原理是什么？(logits / T)
# - T=0 是否完全等价于 greedy decoding？为什么？
# - 高 Temperature 下的"重复惩罚"在哪类任务尤其重要？

import os

USE_MOCK = os.environ.get("USE_REAL_API") != "1"


def _mock_completions(prompt: str, temperature: float, n: int):
    """模拟 OpenAI 返回：temperature 越高，输出差异越大。"""
    import random
    rng = random.Random(int(temperature * 1000))
    base = ["落叶飘黄", "金风送爽", "枫红如火", "稻浪翻涌", "凉意渐浓"]
    if temperature == 0:
        return [base[0]] * n
    return rng.sample(base, n)


def compare_temperatures(prompt: str, temps=None):
    """对比不同 Temperature 下的输出差异"""
    if temps is None:
        temps = [0.0, 0.5, 1.0]
    results = {}
    for t in temps:
        if USE_MOCK:
            results[t] = _mock_completions(prompt, t, n=3)
        else:
            import openai
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=t,
                n=3,  # 每个温度生成3个样本
            )
            results[t] = [c.message.content for c in response.choices]
    return results


if __name__ == "__main__":
    # 示例：Temperature=0 时三次输出完全相同；Temperature=1 时三次输出各不相同
    results = compare_temperatures("用一句话形容秋天", [0.0, 0.7, 1.2])
    for t, samples in results.items():
        print(f"\n--- Temperature={t} ---")
        for i, s in enumerate(samples, 1):
            print(f"  样本{i}: {s}")
    print("\nOK")
