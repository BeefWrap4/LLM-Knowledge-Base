# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.6.4 企业级AI治理框架
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 09_ai_governance_framework.py
# expected_runtime: <1s
# expected_output: AI系统注册/影响评估/模型卡片演示 + "OK"
# ---
# See: ../tutorial/23_AI安全与伦理.md#2364-企业级ai治理框架
# Interview hooks:
#   1. 企业级AI治理框架应该包含哪些核心组件？
#   2. EU AI Act的高风险AI系统需要进行哪些评估？
#   3. 模型卡片（Model Card）应包含哪些关键字段？
"""
企业级AI治理框架的核心组件（架构示例）

面试要点：
1. AI治理不是一次性活动，而是持续生命周期
2. 需要跨部门协作（技术+法务+合规+业务）
"""

from datetime import datetime
from enum import Enum


class AIRiskLevel(Enum):
    """AI系统风险等级（参考EU AI Act）"""

    MINIMAL = "最低风险"
    LIMITED = "有限风险"
    HIGH = "高风险"
    UNACCEPTABLE = "不可接受"


class AIGovernanceFramework:
    """企业AI治理框架

    覆盖六大治理维度：
    1. 策略与政策
    2. 风险评估
    3. 合规审查
    4. 监控与审计
    5. 培训与意识
    6. 事件响应
    """

    def __init__(self):
        self.ai_inventory = []  # AI系统清单
        self.policies = {}  # 治理政策
        self.audit_log = []  # 审计日志

    def register_ai_system(
        self,
        name: str,
        description: str,
        use_case: str,
        model_info: dict,
        data_sources: list[str],
        risk_level: AIRiskLevel,
    ) -> str:
        """注册AI系统到治理清单

        每个AI系统上线前必须完成注册，
        记录元数据、风险评估、合规状态。
        """
        system = {
            "name": name,
            "description": description,
            "use_case": use_case,
            "model_info": model_info,
            "data_sources": data_sources,
            "risk_level": risk_level,
            "registered_at": datetime.now().isoformat(),
            "approved": False,
            "last_review": None,
            "compliance_status": {},
        }
        self.ai_inventory.append(system)
        return f"System '{name}' registered with risk level: {risk_level.value}"

    def conduct_impact_assessment(self, system_name: str) -> dict:
        """AI影响评估（参考EU AI Act高风险要求）

        面试常问：什么情况需要做影响评估？
        - 涉及基本权利（就业、教育、信贷）
        - 涉及公共安全
        - 涉及大规模数据收集
        """
        assessment = {
            "system": system_name,
            "assessed_at": datetime.now().isoformat(),
            "dimensions": {
                "fundamental_rights": "需要评估对隐私、非歧视等基本权利的影响",
                "safety": "需要评估对人身安全、公共安全的潜在风险",
                "fairness": "需要评估对不同群体的差异化影响",
                "transparency": "需要评估用户是否能理解AI决策",
                "accountability": "需要明确AI决策的责任归属",
            },
            "mitigation_required": [],
        }
        return assessment

    def generate_model_card(self, system_name: str) -> dict:
        """生成模型卡片（Model Card）

        模型卡片是AI透明度的关键工具，
        参考Google Model Cards for Model Reporting框架。
        """
        system = next((s for s in self.ai_inventory if s["name"] == system_name), None)
        if not system:
            return {"error": f"System '{system_name}' not found in inventory"}

        return {
            "model_details": {
                "name": system["name"],
                "version": system["model_info"].get("version", "N/A"),
                "type": system["model_info"].get("type", "N/A"),
                "release_date": datetime.now().isoformat(),
            },
            "intended_use": {
                "primary_use_case": system["use_case"],
                "out_of_scope_uses": [],
                "intended_users": system["model_info"].get("users", []),
            },
            "performance": {"metrics": {}, "evaluation_data": {}, "limitations": []},
            "ethical_considerations": {
                "bias_assessment": "待补充",
                "privacy_analysis": "待补充",
                "safety_testing": "待补充",
            },
            "recommendations": {
                "monitoring": "建议每月审查",
                "human_oversight": "高风险系统需要人工审核",
            },
        }


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=== 企业AI治理框架演示 ===")

    framework = AIGovernanceFramework()

    # 1. 注册一个高风险AI系统（模拟贷款审批）
    registration = framework.register_ai_system(
        name="LoanApprovalBot",
        description="自动化贷款审批AI系统",
        use_case="消费信贷审批",
        model_info={"version": "v1.2.0", "type": "XGBoost+LLM", "users": ["银行客户经理"]},
        data_sources=["交易历史", "征信报告", "用户画像"],
        risk_level=AIRiskLevel.HIGH,
    )
    print(f"\n[1] 系统注册: {registration}")

    # 2. 影响评估
    assessment = framework.conduct_impact_assessment("LoanApprovalBot")
    print("\n[2] 影响评估维度:")
    for dim, desc in assessment["dimensions"].items():
        print(f"  - {dim}: {desc}")

    # 3. 生成模型卡片
    card = framework.generate_model_card("LoanApprovalBot")
    print("\n[3] 模型卡片生成:")
    for section, content in card.items():
        if isinstance(content, dict):
            print(f"  [{section}]:")
            for k, v in content.items():
                print(f"    {k}: {v}")
        else:
            print(f"  [{section}]: {content}")

    # 4. 治理清单汇总
    print(f"\n[4] AI系统清单: {len(framework.ai_inventory)}个系统已注册")
    print("OK")
