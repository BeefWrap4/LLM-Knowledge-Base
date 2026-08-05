# ---
# chapter: 37
# topic: RAG、Agent 与安全评估
# topic_id: rag_indexing.rag_evaluator
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (none — stdlib only, real LLM optional)
# run: python 25_rag_evaluator.py
# expected_runtime: <1s (mock mode)
# expected_output: faithfulness and relevance scores
# ---
# See: ../../../37_RAG_Agent与安全评估.md
# Interview hooks:
#   1. LLM-as-Judge 评估的偏差来源有哪些（位置偏好、长度偏好、自我偏好）？
#   2. Faithfulness 和 Answer Relevance 评估的关键差异是什么？
#   3. 如何构建小规模（50-100）但高质量的 RAG 评估集？

import json
import os
import re

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6")
OPENAI_REASONING_KWARGS = (
    {"reasoning_effort": "none"} if DEFAULT_OPENAI_MODEL.startswith("gpt-5.6") else {}
)


def _simple_tokens(text: str) -> set[str]:
    """仅用于离线 smoke 的中英文字符/词元集合，不替代正式 RAG 评估。"""

    return set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text.lower()))


class RAGEvaluator:
    """RAG 评估器：使用 LLM 作为裁判（mock 演示版）"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def evaluate_faithfulness(self, answer: str, contexts: list) -> float:
        """
        评估回答的忠实度（Faithfulness）
        检查回答中的每个陈述是否都能在上下文中找到依据
        """
        context_text = "\n".join(contexts)
        prompt = f"""评估以下回答是否忠实于提供的上下文。

上下文：
{context_text}

回答：{answer}

请逐句分析回答中的每个事实性陈述，判断是否能从上下文中找到依据。
输出格式：
{{
    "faithfulness_score": 0-1之间的浮点数,
    "violations": ["未找到依据的陈述1", "未找到依据的陈述2"]
}}"""

        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **OPENAI_REASONING_KWARGS,
            )
            content = response.choices[0].message.content
            json_match = re.search(r"\{.*?\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("faithfulness_score", 0.0)
            return 0.0
        # Mock: 仅用字符/词元覆盖率做确定性 smoke，不声称等价于 LLM-as-Judge。
        if not contexts:
            return 0.0
        context_words = _simple_tokens(" ".join(contexts))
        answer_words = _simple_tokens(answer)
        coverage = len(answer_words & context_words) / max(len(answer_words), 1)
        return float(round(coverage, 3))

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """评估回答的相关性"""
        prompt = f"""评估以下回答是否与问题相关。

问题：{question}
回答：{answer}

如果回答完全跑题，输出 0；如果完全切题，输出 1。
只输出一个 0-1 之间的数字。"""

        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **OPENAI_REASONING_KWARGS,
            )
            try:
                return float(response.choices[0].message.content.strip())
            except ValueError:
                return 0.5
        # Mock: 关键词重合度作为相关性近似
        q_words = _simple_tokens(question)
        a_words = _simple_tokens(answer)
        return float(round(len(q_words & a_words) / max(len(q_words | a_words), 1), 3))

    def evaluate(self, question: str, answer: str, contexts: list) -> dict:
        """完整评估"""
        f = self.evaluate_faithfulness(answer, contexts)
        r = self.evaluate_answer_relevance(question, answer)
        return {
            "faithfulness": f,
            "relevance": r,
            "overall": round(0.7 * f + 0.3 * r, 3),
            "mode": "llm_judge" if self.llm is not None else "offline_heuristic_smoke",
        }


if __name__ == "__main__":
    evaluator = RAGEvaluator(llm_client=None)
    question = "公司的年假政策是什么？"
    answer = "员工每年享有 15 天带薪年假，需提前申请。"
    contexts = ["员工每年享有 15 天带薪年假。", "请假需提前申请。"]
    out = evaluator.evaluate(question, answer, contexts)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("OK")
