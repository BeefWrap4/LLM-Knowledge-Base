# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.7.3 金丝雀发布与回滚策略
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (stdlib only)
# run: python 18_canary_controller.py
# expected_runtime: < 1s
# expected_output: Canary status JSON with stage, traffic split, error rate
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2073-金丝雀发布与回滚策略-⭐⭐⭐
# Interview hooks:
#  - 金丝雀发布 vs A/B 测试 vs 蓝绿部署的差别与适用场景？
#  - 自动回滚的触发条件应该写"立即"还是"持续 N 分钟"？
#  - 健康检查聚合（成功率/错误率/延迟）怎么加权？

import json
import time
from dataclasses import dataclass, field
from enum import Enum


class ReleaseStage(Enum):
    CANARY_5 = 0.05
    CANARY_25 = 0.25
    CANARY_50 = 0.50
    FULL = 1.0


@dataclass
class CanaryController:
    """金丝雀发布控制器"""

    new_version: str
    old_version: str
    current_stage: ReleaseStage = ReleaseStage.CANARY_5
    stage_start_time: float = field(default_factory=time.time)
    health_checks_passed: int = 0
    health_checks_total: int = 0
    auto_rollback_enabled: bool = True

    def get_traffic_split(self) -> float:
        return self.current_stage.value

    def should_promote(self) -> bool:
        elapsed_minutes = (time.time() - self.stage_start_time) / 60
        error_rate = 1 - self.health_checks_passed / max(self.health_checks_total, 1)

        min_minutes = {
            ReleaseStage.CANARY_5: 30,
            ReleaseStage.CANARY_25: 60,
            ReleaseStage.CANARY_50: 60,
        }
        return (
            elapsed_minutes >= min_minutes.get(self.current_stage, 30)
            and error_rate < 0.005
            and self.current_stage != ReleaseStage.FULL
        )

    def promote(self):
        stages = list(ReleaseStage)
        idx = stages.index(self.current_stage)
        if idx < len(stages) - 1:
            self.current_stage = stages[idx + 1]
            self.stage_start_time = time.time()
            print(f"🚀 推进到 {self.current_stage.name}: {self.current_stage.value * 100:.0f}% 流量")

    def rollback(self, reason: str):
        self.current_stage = ReleaseStage.CANARY_5
        self.stage_start_time = time.time()
        print(f"⏪ 回滚到 {self.old_version}！原因：{reason}")

    def record_health_check(self, passed: bool):
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
            "error_rate": 1 - self.health_checks_passed / max(self.health_checks_total, 1),
        }


if __name__ == "__main__":
    controller = CanaryController(
        new_version="prompt_v4.0.0",
        old_version="prompt_v3.2.0",
    )
    # 模拟金丝雀发布
    for i in range(50):
        controller.record_health_check(i < 48)
        if controller.should_promote():
            controller.promote()
        if controller.auto_rollback_enabled and controller.get_status()["error_rate"] > 0.01:
            controller.rollback("错误率超过 1% 阈值")
            break
    print(json.dumps(controller.get_status(), ensure_ascii=False, indent=2))
