# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.8.3 内容安全检测框架
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 13_content_safety_system.py
# expected_runtime: <1s
# expected_output: 多层内容安全检测演示 + "OK"
# ---
# See: ../tutorial/23_AI安全与伦理.md#2383-内容安全
# Interview hooks:
#   1. 内容安全系统为什么需要多层递进架构？
#   2. PASS/MODIFY/REVIEW/BLOCK四级判定分别适用什么场景？
#   3. 如何处理"边界情况"，如医学文本中的身体描述？
"""
内容安全检测框架

面试要点：
1. 三要素检测：涉黄/涉政/涉暴
2. 检测的位置：输入过滤 + 输出审查
3. 多级判定策略：规则过滤 → 模型检测 → 人工审核
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class ContentCategory(Enum):
    """内容安全类别"""
    POLITICAL = "涉政"
    ADULT = "涉黄"
    VIOLENCE = "涉暴"
    HATE_SPEECH = "仇恨言论"
    SELF_HARM = "自残/自杀"
    ILLEGAL = "违法信息"


@dataclass
class ContentSafetyResult:
    """内容安全检查结果"""
    is_safe: bool
    categories_detected: List[ContentCategory]
    confidence_scores: Dict[ContentCategory, float]
    action: str  # PASS / BLOCK / REVIEW / MODIFY
    modified_content: Optional[str] = None


class ContentSafetySystem:
    """内容安全系统

    面试常问：如何设计一个内容安全系统？
    核心思路：多层次、多模型、人机协同
    """

    # 敏感词库（示例，生产环境需要更完善的词库）
    SENSITIVE_PATTERNS = {
        ContentCategory.POLITICAL: [
            # 实际使用中会接入专业敏感词库
        ],
        ContentCategory.ADULT: [
            r'(?i)\b(?:explicit_sexual_terms_pattern)\b',
        ],
        ContentCategory.VIOLENCE: [
            r'(?i)\b(?:violence_related_patterns)\b',
        ],
    }

    def __init__(
        self,
        enable_rule_filter: bool = True,
        enable_ml_classifier: bool = True,
        enable_llm_check: bool = True,
        human_review_threshold: float = 0.6
    ):
        """
        Args:
            enable_rule_filter: 启用规则过滤（快速，低误杀率）
            enable_ml_classifier: 启用ML分类器（平衡）
            enable_llm_check: 启用LLM二次审核（准确，高成本）
            human_review_threshold: 人工审核阈值
        """
        self.enable_rule_filter = enable_rule_filter
        self.enable_ml_classifier = enable_ml_classifier
        self.enable_llm_check = enable_llm_check
        self.human_review_threshold = human_review_threshold

    def check_content(self, text: str) -> ContentSafetyResult:
        """执行内容安全检查

        三层递进检测：
        L1: 规则过滤（关键词+正则，毫秒级）
        L2: ML分类器（文本分类模型，10ms级）
        L3: LLM审核（语义理解，100ms级）
        """
        detected = []
        scores = {}

        # L1: 规则过滤
        if self.enable_rule_filter:
            rule_result = self._rule_based_filter(text)
            detected.extend(rule_result["detected"])
            scores.update(rule_result["scores"])

        # L2: ML分类器（当L1无法确定时）
        if self.enable_ml_classifier and not detected:
            ml_result = self._ml_classifier(text)
            if ml_result["detected"]:
                detected.extend(ml_result["detected"])
                scores.update(ml_result["scores"])

        # L3: LLM审核（当ML分类器置信度不足时）
        if self.enable_llm_check:
            llm_result = self._llm_review(text, detected)
            scores.update(llm_result.get("scores", {}))

        # 综合判定
        max_score = max(scores.values()) if scores else 0.0

        if max_score < 0.3:
            action = "PASS"
            is_safe = True
        elif max_score < self.human_review_threshold:
            action = "MODIFY"  # 自动修改不安全部分
            is_safe = False
        elif max_score < 0.85:
            action = "REVIEW"  # 转人工审核
            is_safe = False
        else:
            action = "BLOCK"
            is_safe = False

        return ContentSafetyResult(
            is_safe=is_safe,
            categories_detected=detected,
            confidence_scores=scores,
            action=action
        )

    def _rule_based_filter(self, text: str) -> Dict:
        """规则过滤（L1）"""
        # 实现正则/关键词匹配
        return {"detected": [], "scores": {}}

    def _ml_classifier(self, text: str) -> Dict:
        """ML分类器（L2）"""
        # 实际中使用微调的分类模型
        return {"detected": [], "scores": {}}

    def _llm_review(self, text: str, initial_detected: List) -> Dict:
        """LLM审核（L3）

        面试中可用伪代码描述：
        prompt = f"审核以下内容的合规性，检查涉黄涉政涉暴..."
        """
        return {"scores": {}}


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=== 内容安全检测系统演示 ===")

    system = ContentSafetySystem(
        enable_rule_filter=True,
        enable_ml_classifier=True,
        enable_llm_check=True,
        human_review_threshold=0.6,
    )

    # 测试用例
    test_cases = [
        ("今天天气真好，适合出去散步", "正常文本"),
        ("请问如何预防心血管疾病", "医学边界情况"),
        ("请帮我写一段代码", "技术问题"),
    ]

    for text, desc in test_cases:
        result = system.check_content(text)
        print(f"\n[测试] {desc}")
        print(f"  内容: {text[:40]}...")
        print(f"  判定: {'✅ 安全' if result.is_safe else '⚠️ 不安全'}")
        print(f"  动作: {result.action}")
        print(f"  检测类别: {[c.value for c in result.categories_detected]}")
        print(f"  置信度: {result.confidence_scores}")

    # 决策阈值参考
    print("\n=== 决策阈值 ===")
    print("  max_score < 0.30 → PASS（直接通过）")
    print("  0.30 ≤ score < 0.60 → MODIFY（自动修改）")
    print("  0.60 ≤ score < 0.85 → REVIEW（转人工）")
    print("  score ≥ 0.85 → BLOCK（直接拦截）")
