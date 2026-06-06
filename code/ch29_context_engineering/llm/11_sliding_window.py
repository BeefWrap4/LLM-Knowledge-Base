# ---
# chapter: 29
# topic: Sliding Window 策略 — 保留最近 K 轮, 简单但有效
# section: 29.4.1
# difficulty: ⭐⭐⭐
# tier: llm
# deps: 无
# run: python 11_sliding_window.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.4.1
# Cross-refs:
#   - Ch15 Agent (turn 管理)
#   - Ch18 LangChain ConversationBufferWindowMemory
#   - Ch07 Compaction 对比
#
# Interview hooks:
#   - "Sliding Window 优缺点?"   →  简单/快; 但会丢早期信息
#   - "Window K 选多大?"         →  任务而定, 简单 QA 用 4-6, 复杂任务用 10-20
#   - "Window vs Compaction?"    →  Window 简单粗暴; Compaction 保留语义摘要

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SlidingWindowMemory:
    """保留最近 K 轮 (user+assistant 算 1 轮) 的对话, 旧的自动出队。"""
    k_turns: int = 5
    _turns: deque = field(default_factory=deque)

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append({"user": user, "assistant": assistant})
        while len(self._turns) > self.k_turns:
            self._turns.popleft()

    def to_messages(self, system: str = "") -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for t in self._turns:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        return msgs

    def token_estimate(self) -> int:
        n = sum(len(t["user"]) + len(t["assistant"]) for t in self._turns) // 2
        return n

    def dropped(self) -> int:
        return max(0, len(self._turns) - self.k_turns)


def compare_policies() -> None:
    """对比 Sliding Window 与 Compaction 的取舍。"""
    import json
    long_history = [
        ("我叫 Alice, 喜欢科幻片", "已记录"),
        ("我住在北京", "已记录"),
        ("我是工程师", "已记录"),
        ("推荐电影?", "《星际穿越》"),
        ("再推荐?", "《盗梦空间》"),
        ("第三个?", "《银翼杀手》"),
        ("这个剧情?", "讲述..."),
        ("导演是谁?", "诺兰"),
        ("还有谁?", "诺兰/维伦纽瓦/斯科特"),
        ("我要看科幻喜剧", "《银河系漫游指南》"),
    ]
    print(f"=== 长对话 ({len(long_history)} 轮) ===\n")
    sw = SlidingWindowMemory(k_turns=3)
    for u, a in long_history:
        sw.add_turn(u, a)
    print(f"SlidingWindow(K=3) 保留消息数: {len(sw.to_messages())}  (含 system)")
    print(f"  估算 tokens: {sw.token_estimate()}")
    print(f"  早期信息 ('我喜欢科幻片', '北京', '工程师') 是否还在: ",
          "科幻片" in json.dumps(sw.to_messages(), ensure_ascii=False))

    print("\n→ Sliding Window 优点: 实现简单, O(1) 插入")
    print("→ 缺点: 早期 '我喜欢科幻片' 等关键事实已丢失, 后续推荐会受影响")


def run_demo() -> None:
    compare_policies()


if __name__ == "__main__":
    run_demo()
    print("\nOK")
