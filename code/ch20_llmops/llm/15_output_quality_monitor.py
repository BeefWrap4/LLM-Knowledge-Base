# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.6.3 输出质量监控
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 15_output_quality_monitor.py
# expected_runtime: < 1s
# expected_output: Hallucination risk, safety flags, format compliance dicts
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2063-输出质量监控-⭐⭐⭐
# Interview hooks:
#  - 启发式幻觉检测与 LLM-as-Judge 各有什么取舍？
#  - PII 检测正则为什么不能覆盖全部场景（命名实体、隐式 PII）？
#  - JSON 格式合规性检查的常见边界条件（嵌套、引号转义）？

import json as json_module
import re


class OutputQualityMonitor:
    """LLM 输出质量自动检测器"""

    @staticmethod
    def check_hallucination_indicators(response: str, context: str | None = None) -> dict:
        """启发式幻觉检测"""
        indicators = {
            "excessive_confidence": False,
            "unverifiable_claims": False,
            "contradiction": False,
            "hallucination_risk": "low",
        }

        overconfident_phrases = [
            "毫无疑问",
            "绝对是",
            "100%确定",
            "一定是",
            "definitely",
            "absolutely",
            "without any doubt",
        ]
        for phrase in overconfident_phrases:
            if phrase in response:
                indicators["excessive_confidence"] = True
                break

        if context:
            # 简化：生产中应做实体/NLI 校验
            pass

        if "但是" in response and "因此" in response:
            indicators["hallucination_risk"] = "medium"

        risk_score = sum(
            [
                indicators["excessive_confidence"],
                indicators["unverifiable_claims"],
                indicators["contradiction"],
            ]
        )
        if risk_score >= 2:
            indicators["hallucination_risk"] = "high"
        return indicators

    @staticmethod
    def check_safety(response: str) -> dict:
        """安全检查（简化版）"""
        safety_flags = {
            "harmful_content": False,
            "pii_leak": False,
            "prompt_injection_reflected": False,
            "overall_safe": True,
        }
        pii_patterns = {
            "phone": r"\b1[3-9]\d{9}\b",
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            "id_card": r"\b\d{17}[\dXx]\b",
        }
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, response):
                safety_flags["pii_leak"] = True
                safety_flags["overall_safe"] = False
                break
        return safety_flags

    @staticmethod
    def check_format_compliance(response: str, expected_format: str = "json") -> dict:
        """格式合规检查（支持 ```json ``` 包裹）"""
        result = {"format": expected_format, "compliant": False, "error": None}
        if expected_format == "json":
            try:
                if "```json" in response:
                    start = response.index("```json") + 7
                    end = response.index("```", start)
                    json_str = response[start:end].strip()
                elif "{" in response:
                    start = response.index("{")
                    end = response.rindex("}") + 1
                    json_str = response[start:end]
                else:
                    json_str = response
                json_module.loads(json_str)
                result["compliant"] = True
            except (ValueError, json_module.JSONDecodeError) as e:
                result["error"] = str(e)
        return result


if __name__ == "__main__":
    monitor = OutputQualityMonitor()

    response = "毫无疑问，Python 是世界上最完美的编程语言，没有任何缺点。"
    hallucination = monitor.check_hallucination_indicators(response)
    print(f"幻觉风险: {hallucination['hallucination_risk']}")

    safety = monitor.check_safety("我的手机是 13800000000，邮箱 a@b.com")
    print(f"安全检查: {safety}")

    fmt = monitor.check_format_compliance('```json\n{"a": 1}\n```')
    print(f"格式合规: {fmt}")
    print("OK")
