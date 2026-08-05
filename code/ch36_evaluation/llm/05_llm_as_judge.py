# ---
# chapter: 36
# topic: 大模型评估基础
# topic_id: evaluation.llm_as_judge
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai
# run: python 05_llm_as_judge.py
# expected_runtime: <2s (mock mode)
# expected_output: Pointwise and pairwise Judge results followed by OK
# ---
# See: ../../../36_大模型评估基础.md
# Interview hooks:
# - What biases does LLM-as-Judge suffer from and how do you mitigate them?
# - Pointwise scoring vs Pairwise comparison: which is more stable?
# - How do you design a Judge prompt to minimize position bias?

"""LLM-as-Judge 示例。

默认 ``LLM_MOCK=1``，不会读取密钥、导入 OpenAI SDK 或访问网络。仅当显式设置
``LLM_MOCK=0`` 时，才使用 Responses API 和结构化输出调用 ``OPENAI_MODEL``
（默认 ``gpt-5.6``）。真实模式中的缺少依赖、缺少凭据和解析错误都会直接报错，
不会伪装成 mock 成功。
"""

import json
import os
from typing import Any

POINTWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "dimensions": {
            "type": "object",
            "properties": {
                "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
                "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
                "helpfulness": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["accuracy", "completeness", "clarity", "helpfulness"],
            "additionalProperties": False,
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "justification": {"type": "string"},
    },
    "required": [
        "overall_score",
        "dimensions",
        "strengths",
        "weaknesses",
        "justification",
    ],
    "additionalProperties": False,
}

PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "key_difference": {"type": "string"},
    },
    "required": ["winner", "confidence", "reasoning", "key_difference"],
    "additionalProperties": False,
}


class LLMJudge:
    """支持 Pointwise 与 Pairwise 的 OpenAI Judge。"""

    def __init__(
        self,
        judge_model: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.judge_model = judge_model or os.environ.get("OPENAI_MODEL", "gpt-5.6")
        self.mock_mode = os.environ.get("LLM_MOCK", "1") != "0"
        self.client = client

        if self.mock_mode or self.client is not None:
            return

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("真实模式需要 OPENAI_API_KEY；默认请使用 LLM_MOCK=1")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("真实模式需要安装 openai：pip install openai") from exc

        self.client = OpenAI()

    @staticmethod
    def build_prompt(question: str, answer: str, rubric: str) -> str:
        """构建单点评分 prompt。"""
        return f"""请根据评分标准评审 AI 回答。不得补写回答中不存在的事实。

【问题】
{question}

【AI 回答】
{answer}

【评分标准】
{rubric}

请给出总体分、各维度分、优点、不足和简要理由。"""

    def _request_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("OpenAI client 未初始化")

        response = self.client.responses.create(
            model=self.judge_model,
            instructions=(
                "你是独立的 AI 质量评审员。严格按 rubric 评分；不要因答案更长、位置靠前或风格更像自己而加分。"
            ),
            input=prompt,
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        if not response.output_text:
            raise ValueError("Judge 返回了空的结构化输出")
        return json.loads(response.output_text)

    def evaluate(
        self,
        question: str,
        answer: str,
        rubric: str | None = None,
    ) -> dict[str, Any]:
        """执行单点评分。"""
        rubric = (
            rubric
            or """评分维度（每项 1-5 分）：
1. 准确性：回答是否事实正确
2. 完整性：是否覆盖问题要求
3. 清晰性：表达是否清晰、逻辑是否连贯
4. 有用性：回答对用户是否有实际帮助"""
        )

        if self.mock_mode:
            return {
                "overall_score": 4,
                "dimensions": {
                    "accuracy": 5,
                    "completeness": 4,
                    "clarity": 5,
                    "helpfulness": 4,
                },
                "strengths": ["概念解释清晰", "提供了具体算法"],
                "weaknesses": ["可补充其他探索策略"],
                "justification": "[mock 示意值，非模型实测] 回答覆盖了核心概念。",
            }

        return self._request_json(
            prompt=self.build_prompt(question, answer, rubric),
            schema_name="pointwise_judgement",
            schema=POINTWISE_SCHEMA,
        )

    def pairwise_compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> dict[str, Any]:
        """比较两个回答；生产评估还应交换 A/B 位置并检查一致性。"""
        if self.mock_mode:
            return {
                "winner": "A",
                "confidence": 0.75,
                "reasoning": "[mock 示意值，非模型实测] A 更完整。",
                "key_difference": "A 给出了具体算法。",
            }

        prompt = f"""比较两个回答；只依据准确性、完整性、清晰性和有用性判断。

【问题】
{question}

【回答 A】
{answer_a}

【回答 B】
{answer_b}

返回 winner、confidence、reasoning、key_difference。正式评测应再交换 A/B 位置复评。"""
        return self._request_json(
            prompt=prompt,
            schema_name="pairwise_judgement",
            schema=PAIRWISE_SCHEMA,
        )


if __name__ == "__main__":
    judge = LLMJudge()
    question = "请解释强化学习中的探索与利用权衡。"
    answer_a = (
        "探索用于尝试未知动作，利用用于选择当前最优动作。例如 ε-贪心以 ε 的概率探索，以 1-ε 的概率利用。"
    )
    answer_b = "探索与利用是强化学习中的两个概念，需要平衡。"

    print(json.dumps(judge.evaluate(question, answer_a), ensure_ascii=False, indent=2))
    print(
        json.dumps(
            judge.pairwise_compare(question, answer_a, answer_b),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"mode={'mock' if judge.mock_mode else 'real'}, model={judge.judge_model}")
    print("OK")
