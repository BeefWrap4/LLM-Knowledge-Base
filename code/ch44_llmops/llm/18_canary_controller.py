# ---
# chapter: 44
# topic: LLMOps 生命周期与持续交付
# topic_id: llmops.canary_controller
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 18_canary_controller.py
# expected_runtime: < 1s
# expected_output: Canary status JSON with stage, traffic split, error rate
# ---
# See: ../../../44_LLMOps生命周期与持续交付.md
# Interview hooks:
#  - 金丝雀发布 vs A/B 测试 vs 蓝绿部署的差别与适用场景？
#  - 自动回滚的触发条件应该写"立即"还是"持续 N 分钟"？
#  - 健康检查聚合（成功率/错误率/延迟）怎么加权？

import json
import time
from dataclasses import dataclass, field
from enum import Enum


class ReleaseStage(Enum):
    ROLLED_BACK = 0.0
    CANARY_5 = 0.05
    CANARY_25 = 0.25
    CANARY_50 = 0.50
    FULL = 1.0


@dataclass
class CanaryController:
    """金丝雀发布控制器；流量阶段和阈值是可替换的教学策略。"""

    new_version: str
    old_version: str
    promotion_max_error_rate: float
    rollback_error_rate: float
    stage_min_minutes: dict[ReleaseStage, float]
    min_health_checks_per_stage: int
    current_stage: ReleaseStage = ReleaseStage.CANARY_5
    stage_start_time: float = field(default_factory=time.time)
    health_checks_passed: int = 0
    health_checks_total: int = 0
    auto_rollback_enabled: bool = True
    rollback_reason: str | None = None

    def __post_init__(self):
        if not 0 <= self.promotion_max_error_rate < self.rollback_error_rate <= 1:
            raise ValueError("expected 0 <= promotion threshold < rollback threshold <= 1")
        if self.min_health_checks_per_stage <= 0:
            raise ValueError("min_health_checks_per_stage must be positive")
        required_stages = {
            ReleaseStage.CANARY_5,
            ReleaseStage.CANARY_25,
            ReleaseStage.CANARY_50,
        }
        if not required_stages.issubset(self.stage_min_minutes):
            raise ValueError("stage_min_minutes must configure every canary stage")
        if any(minutes < 0 for minutes in self.stage_min_minutes.values()):
            raise ValueError("stage durations must be non-negative")

    def get_traffic_split(self) -> float:
        return self.current_stage.value

    def get_error_rate(self) -> float | None:
        if self.health_checks_total == 0:
            return None
        return 1 - self.health_checks_passed / self.health_checks_total

    def should_promote(self) -> bool:
        if self.current_stage in {ReleaseStage.ROLLED_BACK, ReleaseStage.FULL}:
            return False
        if self.health_checks_total < self.min_health_checks_per_stage:
            return False
        elapsed_minutes = (time.time() - self.stage_start_time) / 60
        error_rate = self.get_error_rate()

        return (
            elapsed_minutes >= self.stage_min_minutes[self.current_stage]
            and error_rate is not None
            and error_rate < self.promotion_max_error_rate
        )

    def promote(self):
        stages = [
            ReleaseStage.CANARY_5,
            ReleaseStage.CANARY_25,
            ReleaseStage.CANARY_50,
            ReleaseStage.FULL,
        ]
        if self.current_stage not in stages:
            raise RuntimeError("cannot promote a rolled-back release")
        idx = stages.index(self.current_stage)
        if idx < len(stages) - 1:
            self.current_stage = stages[idx + 1]
            self.stage_start_time = time.time()
            self.health_checks_passed = 0
            self.health_checks_total = 0
            print(f"🚀 推进到 {self.current_stage.name}: {self.current_stage.value * 100:.0f}% 流量")

    def rollback(self, reason: str):
        self.current_stage = ReleaseStage.ROLLED_BACK
        self.stage_start_time = time.time()
        self.rollback_reason = reason
        print(f"⏪ 回滚到 {self.old_version}！原因：{reason}")

    def record_health_check(self, passed: bool):
        if self.current_stage == ReleaseStage.ROLLED_BACK:
            return
        self.health_checks_total += 1
        if passed:
            self.health_checks_passed += 1

    def get_status(self) -> dict:
        return {
            "stage": self.current_stage.name,
            "traffic_split": self.current_stage.value,
            "new_version": self.new_version,
            "old_version": self.old_version,
            "elapsed_minutes": (time.time() - self.stage_start_time) / 60,
            "health_checks": self.health_checks_total,
            "error_rate": self.get_error_rate(),
            "rollback_reason": self.rollback_reason,
        }


if __name__ == "__main__":
    controller = CanaryController(
        new_version="prompt_v4.0.0",
        old_version="prompt_v3.2.0",
        # 教学发布策略；生产值来自风险分级、容量与错误预算。
        promotion_max_error_rate=0.005,
        rollback_error_rate=0.01,
        stage_min_minutes={
            ReleaseStage.CANARY_5: 30,
            ReleaseStage.CANARY_25: 60,
            ReleaseStage.CANARY_50: 60,
        },
        min_health_checks_per_stage=20,
    )
    # 模拟金丝雀发布
    for i in range(50):
        controller.record_health_check(i < 48)
        if controller.should_promote():
            controller.promote()
        error_rate = controller.get_error_rate()
        if (
            controller.auto_rollback_enabled
            and controller.health_checks_total >= controller.min_health_checks_per_stage
            and error_rate is not None
            and error_rate > controller.rollback_error_rate
        ):
            controller.rollback("错误率超过配置的回滚阈值")
            break
    print(json.dumps(controller.get_status(), ensure_ascii=False, indent=2))
    print("OK")
