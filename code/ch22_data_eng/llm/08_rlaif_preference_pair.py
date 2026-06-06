# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.5.1 RLHF/RLAIF 数据构建
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 08_rlaif_preference_pair.py
# expected_runtime: <2s
# expected_output: 偏好对字典（prompt, chosen, rejected, scores）
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. RLAIF 为什么需要"交换顺序进行二次评判"？这能解决什么问题？
#   2. 偏好数据中的"位置偏差 (Position Bias)"和"长度偏差 (Length Bias)"如何缓解？
#   3. UltraFeedback 的 4 维度 × 5 级评分体系设计理念是什么？比单一奖励好在哪？

import json
from typing import Dict, Optional

JUDGE_PROMPT = """你是一个公正的 AI 回复评判官。请对以下两个回复进行评估：

用户问题：{prompt}

回复 A：{response_a}

回复 B：{response_b}

评估维度：
1. 帮助性（是否真正解答了问题）
2. 准确性（是否包含事实错误）
3. 完整性（是否覆盖了关键信息）
4. 安全性（是否包含有害内容）

请输出 JSON 格式：
{{
  "winner": "A" | "B" | "tie",
  "reasoning": "<判断理由>",
  "scores_a": {{"helpfulness": 1-10, "accuracy": 1-10, "completeness": 1-10, "safety": 1-10}},
  "scores_b": {{"helpfulness": 1-10, "accuracy": 1-10, "completeness": 1-10, "safety": 1-10}}
}}"""


def _parse_judgment(judgment_str: str) -> Dict:
    """Mock 解析 judge 输出 - 真实场景应使用 json.loads 或 Pydantic"""
    # 这里假设 judge_llm 已经返回了 dict；在独立运行时模拟一份样本
    return {
        "winner": "A",
        "scores_a": {"helpfulness": 8, "accuracy": 9, "completeness": 7, "safety": 10},
        "scores_b": {"helpfulness": 6, "accuracy": 7, "completeness": 5, "safety": 10},
    }


def mock_judge_llm(prompt: str) -> Dict:
    """Mock 一个 LLM Judge - 实际可使用 GPT-4 / Claude / Prometheus"""
    # 在真实场景下，调用 LLM 后用 json.loads 解析
    return _parse_judgment(prompt)


def generate_preference_pair(prompt: str, response_a: str,
                             response_b: str, judge_llm) -> Optional[Dict]:
    """通过 AI Judge 生成偏好数据对，自动处理位置偏差"""
    # 关键：交换顺序进行二次评判，缓解 position bias
    judgment_1 = judge_llm(JUDGE_PROMPT.format(
        prompt=prompt, response_a=response_a, response_b=response_b))
    judgment_2 = judge_llm(JUDGE_PROMPT.format(
        prompt=prompt, response_a=response_b, response_b=response_a))

    # 两次结果一致才保留
    if judgment_1["winner"] == "A" and judgment_2["winner"] == "B":
        chosen, rejected = response_a, response_b
    elif judgment_1["winner"] == "B" and judgment_2["winner"] == "A":
        chosen, rejected = response_b, response_a
    else:
        return None  # 不一致或并列，丢弃

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "confidence": "high",
        "scores": judgment_1["scores_a"] if chosen == response_a else judgment_1["scores_b"]
    }


def main():
    # 演示用法
    prompt = "请解释什么是大语言模型？"
    response_a = "大语言模型是基于 Transformer 架构的大规模预训练模型，通过自监督学习在海量文本上学习语言规律，能够完成文本生成、问答、翻译等任务。"
    response_b = "大语言模型是 AI。"

    pair = generate_preference_pair(prompt, response_a, response_b, mock_judge_llm)
    print(json.dumps(pair, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    print("OK")
