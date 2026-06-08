import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.3.3 LLM-as-Judge 实战
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai
# run: python 05_llm_as_judge.py
# expected_runtime: <2s (mock mode)
# expected_output: Initialization message and pairwise comparison demo
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - What biases does LLM-as-Judge suffer from and how do you mitigate them?
# - Pointwise scoring vs Pairwise comparison: which is more stable?
# - How do you design a Judge prompt to minimize position bias?

"""LLM-as-Judge 完整示例。

使用 GPT-4o 作为评审模型，对候选模型的输出进行多维度评分。
支持 Pointwise（单点打分）和 Pairwise（成对比较）两种范式。
"""
import json
import os


class LLMJudge:
    """
    LLM-as-Judge 评估器

    使用 GPT-4 作为评审模型，对候选模型的输出进行多维度评分。
    """

    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        self.client = None
        self.mock_mode = os.environ.get("LLM_JUDGE_MOCK", "1") == "1"
        if not self.mock_mode:
            # Wave 16: 改用 UnifiedClient (deepseek/kimi/siliconflow/MiniMax)
            try:
                from shared.llm_client import UnifiedClient

                self.client = UnifiedClient(provider="openai" if os.environ.get("OPENAI_API_KEY") else None)
            except ImportError as exc:
                print(f"[mock] UnifiedClient 不可用 ({exc}), 切换 mock 模式")
                self.mock_mode = True

    def build_prompt(self, question: str, answer: str, rubric: str) -> str:
        """构建评估 prompt"""
        return f"""你是一个专业的 AI 回答质量评审员。请根据以下评分标准，对 AI 助手的回答进行评分。

【问题】
{question}

【AI 回答】
{answer}

【评分标准】
{rubric}

【输出要求】
请以 JSON 格式输出评分结果，包含以下字段：
- overall_score: 1-5 的总体评分
- dimensions: 对象，每个维度的分项评分
- strengths: 回答的优点（列表）
- weaknesses: 回答的不足（列表）
- justification: 评分理由的简要说明

只输出 JSON，不要包含其他文字。"""

    def evaluate(self, question: str, answer: str, rubric: str | None = None) -> dict:
        """执行单次评估"""
        if rubric is None:
            rubric = """评分维度（每项 1-5 分）：
1. 准确性 (Accuracy)：回答是否事实正确
2. 完整性 (Completeness)：是否充分回答了问题的所有方面
3. 清晰性 (Clarity)：表达是否清晰、逻辑是否连贯
4. 有用性 (Helpfulness)：回答对用户是否有实际帮助"""

        prompt = self.build_prompt(question, answer, rubric)

        if self.mock_mode:
            return {
                "overall_score": 4,
                "dimensions": {
                    "accuracy": 5,
                    "completeness": 4,
                    "clarity": 5,
                    "helpfulness": 4,
                },
                "strengths": ["概念解释清晰", "比喻生动易懂", "提到具体算法"],
                "weaknesses": ["可补充 UCB、Thompson 采样等方法", "缺少数学表达"],
                "justification": "[mock] 回答准确解释了核心概念，并用生活化比喻帮助理解",
            }

        # Wave 16: 统一接口; 注: response_format 仅 OpenAI 完整支持
        resp = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # 评估任务使用低温度保证一致性
        )
        result = json.loads(resp.content)
        return result

    def pairwise_compare(self, question: str, answer_a: str, answer_b: str) -> dict:
        """
        成对比较：让 Judge 在两个回答中选择更好的一个

        Pairwise comparison 通常比 Pointwise scoring
        更稳定、更接近人类偏好。
        """
        prompt = f"""你是一个专业的 AI 回答质量评审员。请比较以下两个回答，选择更好的一个。

【问题】
{question}

【回答 A】
{answer_a}

【回答 B】
{answer_b}

【输出要求】
请以 JSON 格式输出：
- winner: "A" 或 "B" 或 "tie"
- confidence: 0.0-1.0 置信度
- reasoning: 选择理由
- key_difference: A 和 B 的关键差异

只输出 JSON。"""

        if self.mock_mode:
            return {
                "winner": "A",
                "confidence": 0.75,
                "reasoning": "[mock] 回答 A 在准确性和完整性上更优",
                "key_difference": "[mock] A 提供了更详细的算法解释",
            }

        # Wave 16: 统一接口
        resp = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return json.loads(resp.content)


if __name__ == "__main__":
    # --- 使用示例 ---
    judge = LLMJudge()

    question = "请解释强化学习中的探索与利用权衡。"
    answer = """探索（Exploration）和利用（Exploitation）是强化学习中的核心权衡。
利用是指选择当前已知的最佳动作以获得即时奖励；
探索是指尝试未知的动作以发现可能更好的长期策略。
ε-贪心算法是常用方法：以 ε 的概率随机探索，以 1-ε 的概率贪心利用。
这个权衡类似于"尝试新餐厅"vs"去最喜欢的老餐厅"。"""

    result = judge.evaluate(question, answer)
    print("Pointwise 评估结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Pairwise 比较示例
    answer_b = "探索利用是强化学习概念，二者要平衡。"
    pair_result = judge.pairwise_compare(question, answer, answer_b)
    print("\nPairwise 比较结果:")
    print(json.dumps(pair_result, ensure_ascii=False, indent=2))

    print("LLM-as-Judge 评估框架已就绪（实际调用需有 API Key）")
