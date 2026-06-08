# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.5.2 短期记忆：滑动窗口管理
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 06_sliding_window_memory.py
# expected_runtime: <1s
# expected_output: 滑动窗口会丢弃最早 user/assistant 消息
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.5.2-短期记忆滑动窗口管理
# Interview hooks:
#   1. 滑动窗口管理在估算 token 时常犯的错误是什么？(中英文 token 差异)
#   2. 一定要保留 system message 吗？丢掉后模型人设会怎样？
#   3. 长对话除了滑动窗口还有什么思路？(摘要、关键信息提取)


class SlidingWindowMemory:
    """滑动窗口短期记忆管理"""

    def __init__(self, max_tokens: int = 4000, tokenizer=None):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.tokenizer = tokenizer

    def add(self, role: str, content: str):
        """添加消息，超出窗口时移除最旧的消息"""
        self.messages.append({"role": role, "content": content})
        self._ensure_window_size()

    def _ensure_window_size(self):
        """确保不超过最大 token 数"""
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 2:
            # 保留 system prompt，移除最早的 user/assistant 对话
            if len(self.messages) > 1 and self.messages[1]["role"] != "system":
                self.messages.pop(1)
            else:
                self.messages.pop(2)

    def _estimate_tokens(self) -> int:
        """估算 token 数（粗略估计：1 token ≈ 0.75 中文字符）"""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            # 粗略估算
            total += len(content) // 3 * 2 + 4  # 每条消息 overhead
        return total

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages = []


def main():
    memory = SlidingWindowMemory(max_tokens=4000)
    memory.add("system", "你是一个智能客服助手")
    memory.add("user", "我想退货")
    memory.add("assistant", "好的，请提供您的订单号")
    memory.add("user", "订单号是 #12345")

    print("=== 初始 4 条消息 ===")
    for m in memory.get_messages():
        print(f"  {m['role']:>9} | {m['content']}")

    # 模拟连续追加 60 条长消息，触发窗口收缩
    for i in range(60):
        memory.add("user", f"这是第 {i} 轮用户问题，附带非常非常长的描述文本 " * 8)
        memory.add("assistant", f"第 {i} 轮回答 " * 8)

    msgs = memory.get_messages()
    print(f"\n=== 注入 60 轮后剩余消息数: {len(msgs)} ===")
    print(f"首条: {msgs[0]['role']} -> {msgs[0]['content'][:30]}")
    print(f"末条: {msgs[-1]['role']} -> {msgs[-1]['content'][:30]}")
    assert msgs[0]["role"] == "system", "system 消息必须始终保留"


if __name__ == "__main__":
    main()
