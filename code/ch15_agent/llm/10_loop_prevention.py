# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.8.2 防线1：死循环防范
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 10_loop_prevention.py
# expected_runtime: <1s
# expected_output: 各种死循环模式都被 LoopPrevention 截停
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.8.2-Agent-工程化安全五道防线
# Interview hooks:
#   1. 死循环检测为什么不能只看"是否到了最大步数"？
#   2. A→B→A→B 这种震荡模式怎么识别？窗口长度怎么选？
#   3. 死循环防御的副作用：可能误杀正常重试（连续 create_ticket + verify）。如何折中？

class LoopPrevention:
    """死循环防范机制"""

    def __init__(self, max_steps: int = 10, similarity_threshold: int = 3):
        self.max_steps = max_steps
        self.similarity_threshold = similarity_threshold
        self.action_history: list[str] = []
        self.step_count = 0

    def check(self, action: str) -> tuple[bool, str]:
        """
        检查是否可能陷入死循环

        Returns:
            (是否继续, 原因)
        """
        self.step_count += 1

        # 检查1：最大步数
        if self.step_count > self.max_steps:
            return False, f"超过最大步数限制 ({self.max_steps})"

        # 检查2：相同动作重复
        self.action_history.append(action)
        recent_actions = self.action_history[-self.similarity_threshold:]
        if len(recent_actions) >= self.similarity_threshold:
            if len(set(recent_actions)) == 1:
                return False, f"连续 {self.similarity_threshold} 次执行相同动作"

        # 检查3：动作震荡（A→B→A→B 模式）
        if len(self.action_history) >= 4:
            last4 = self.action_history[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3]:
                return False, "检测到动作震荡模式 (A→B→A→B)"

        return True, "ok"


def main():
    lp = LoopPrevention(max_steps=5, similarity_threshold=3)

    # 场景1：连续执行相同动作 3 次
    print("=== 场景1：连续 3 次相同动作 ===")
    for i in range(5):
        ok, reason = lp.check("search(query='python')")
        print(f"  step {i+1}: ok={ok} | {reason}")
        if not ok:
            break

    # 场景2：震荡模式 A→B→A→B
    print("\n=== 场景2：A→B 震荡 ===")
    lp2 = LoopPrevention(max_steps=20, similarity_threshold=3)
    actions = ["search", "calculator", "search", "calculator", "search"]
    for i, a in enumerate(actions):
        ok, reason = lp2.check(a)
        print(f"  step {i+1} action={a}: ok={ok} | {reason}")
        if not ok:
            break

    # 场景3：超过最大步数
    print("\n=== 场景3：超过最大步数 ===")
    lp3 = LoopPrevention(max_steps=4)
    for i in range(6):
        ok, reason = lp3.check(f"action_{i}")
        print(f"  step {i+1}: ok={ok} | {reason}")
    print("\nOK")


if __name__ == "__main__":
    main()
