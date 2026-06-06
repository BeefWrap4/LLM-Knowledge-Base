# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.8.4 Human-in-the-Loop 设计模式
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 14_human_in_the_loop.py
# expected_runtime: <1s
# expected_output: HITL四种模式演示 + 决策表 + "OK"
# ---
# See: ../tutorial/23_AI安全与伦理.md#2384-human-in-the-loophitl设计模式
# Interview hooks:
#   1. HITL的四种实现模式（前置/后置/异常触发/协同决策）各自的适用场景？
#   2. 如何平衡HITL的延迟、成本与用户体验？
#   3. 异常触发模式中confidence_threshold应如何选择？
"""
Human-in-the-Loop 实现模式

面试中常被问到：
1. 什么场景需要HITL？
2. 如何设计HITL系统？
3. HITL的延迟和成本如何平衡？
"""

from enum import Enum
from typing import Dict, List, Optional


class HITLMode(Enum):
    PRE_REVIEW = "前置审核"      # 先人后机
    POST_REVIEW = "后置审核"     # 先机后人
    EXCEPTION = "异常触发"       # 低置信转人
    COLLABORATIVE = "协同决策"   # 人机协同


class HumanInTheLoop:
    """HITL模式实现框架"""

    def __init__(self, mode: HITLMode, confidence_threshold: float = 0.8):
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def process(self, ai_output, confidence: float) -> Dict:
        """根据HITL模式处理AI输出"""

        if self.mode == HITLMode.PRE_REVIEW:
            # 所有输出先送人工审核
            return {
                "action": "HUMAN_REVIEW",
                "ai_output": ai_output,
                "status": "waiting_approval"
            }

        elif self.mode == HITLMode.POST_REVIEW:
            # AI先输出，异步采样审核
            return {
                "action": "PUBLISH",
                "ai_output": ai_output,
                "status": "will_be_sampled_for_review"
            }

        elif self.mode == HITLMode.EXCEPTION:
            # 高置信自动通过，低置信转人工
            if confidence >= self.confidence_threshold:
                return {
                    "action": "PUBLISH",
                    "ai_output": ai_output,
                    "confidence": confidence,
                    "status": "auto_approved"
                }
            else:
                return {
                    "action": "HUMAN_REVIEW",
                    "ai_output": ai_output,
                    "confidence": confidence,
                    "status": "low_confidence_escalated"
                }

        elif self.mode == HITLMode.COLLABORATIVE:
            # AI提供建议，附带置信度和理由
            return {
                "action": "SUGGEST",
                "ai_output": ai_output,
                "confidence": confidence,
                "instruction": "请人类决策者审核AI建议并做出最终决定"
            }


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=== Human-in-the-Loop 四种模式演示 ===")

    # 演示不同模式的处理逻辑
    sample_output = "AI生成的回复"
    sample_confidence = 0.75

    modes = [
        (HITLMode.PRE_REVIEW, 0.8, "高风险决策（贷款、医疗）"),
        (HITLMode.POST_REVIEW, 0.8, "内容审核、客服回复"),
        (HITLMode.EXCEPTION, 0.8, "推荐系统、分类任务"),
        (HITLMode.COLLABORATIVE, 0.8, "诊断辅助、代码审查"),
    ]

    for mode, threshold, scenario in modes:
        hitl = HumanInTheLoop(mode=mode, confidence_threshold=threshold)
        result = hitl.process(sample_output, sample_confidence)
        print(f"\n[模式] {mode.value}（{scenario}）")
        print(f"  输入: ai_output='{sample_output}', confidence={sample_confidence}")
        print(f"  动作: {result['action']}")
        print(f"  状态: {result.get('status', 'N/A')}")

    # 决策表速查
    print("\n=== HITL模式决策表 ===")
    print("  模式          | 延迟 | 成本 | 适用场景")
    print("  ------------ | ---- | ---- | -----------------------")
    print("  前置审核     | 高   | 高   | 贷款审批、医疗诊断")
    print("  后置审核     | 低   | 中   | 内容审核、客服回复")
    print("  异常触发     | 低*  | 低   | 推荐系统、分类任务")
    print("  协同决策     | 中   | 中   | 诊断辅助、代码审查")
    print("  (* 大部分情况下低延迟)")
    print("OK")
