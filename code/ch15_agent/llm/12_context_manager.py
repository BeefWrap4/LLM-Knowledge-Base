# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.8.2 防线3：上下文污染
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 12_context_manager.py
# expected_runtime: <1s
# expected_output: 长历史被摘要 + 任务分隔符拼接
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.8.2-Agent-工程化安全五道防线
# Interview hooks:
#   1. 上下文污染和上下文窗口溢出是一回事吗？
#   2. 摘要替代早期历史会不会丢失关键信息？怎么校验摘要质量？
#   3. 任务分隔符（task_separator）在多用户多任务场景下的工程意义？


class ContextManager:
    """上下文管理 - 防止污染"""

    def __init__(self, max_context_turns: int = 6):
        self.max_context_turns = max_context_turns
        self.task_separator = "\n--- 新任务 ---\n"

    def build_prompt(self, current_task: str, history: list[dict]) -> str:
        """
        构建干净的 Prompt
        1. 只保留最近 N 轮对话
        2. 不同任务之间加明确分隔
        3. 定期总结历史，替代原始对话
        """
        # 保留最近 N 轮
        recent_history = history[-self.max_context_turns * 2 :]

        # 如果历史很长，用摘要替代早期对话
        if len(history) > self.max_context_turns * 2:
            early_history = history[: -self.max_context_turns * 2]
            summary = self._summarize(early_history)
            context = [summary] + recent_history
        else:
            context = recent_history

        return self._format_prompt(current_task, context)

    def _summarize(self, history: list[dict]) -> dict:
        """对早期历史进行摘要（实际中调用 LLM）"""
        return {
            "role": "system",
            "content": f"[历史摘要] 已完成 {len(history) // 2} 轮交互，关键结论：...",
        }

    def _format_prompt(self, task: str, context: list[dict]) -> str:
        parts = []
        for msg in context:
            parts.append(f"{msg['role']}: {msg['content']}")
        return self.task_separator + f"当前任务: {task}\n" + "\n".join(parts)


def main():
    cm = ContextManager(max_context_turns=3)

    # 模拟 20 轮历史（10 个 user/assistant 对）
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"问题 {i}"})
        history.append({"role": "assistant", "content": f"回答 {i}"})

    prompt = cm.build_prompt("查北京天气", history)

    print("=== 构建的 Prompt ===")
    print(prompt)

    # 校验：早期历史被替换为摘要
    assert "[历史摘要]" in prompt
    assert "问题 0" not in prompt  # 最早的消息被替换
    assert "问题 19" in prompt or "问题 17" in prompt  # 最近几条保留
    print("\nOK")


if __name__ == "__main__":
    main()
