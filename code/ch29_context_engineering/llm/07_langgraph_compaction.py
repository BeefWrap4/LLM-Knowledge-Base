# ---
# chapter: 29
# topic: Context Compaction — 历史消息压缩, 缓解 Context Rot
# section: 29.4
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 07_langgraph_compaction.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.4
# Cross-refs:
#   - Ch15 Agent (token 预算)
#   - Ch18 LangGraph (summarization node)
#   - Ch20 LLMOps (cost control)
#
# Interview hooks:
#   - "Compaction vs Sliding Window?"  →  Compaction 抽取关键事实, 滑动窗口可能丢早期信息
#   - "何时触发 Compaction?"          →  token > 窗口 X% (如 70%) 或消息数 > N
#   - "Compaction 保留什么?"          →  用户偏好/关键事实/最近 N 轮 + 摘要

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompactionPolicy:
    max_tokens: int = 4_000  # 触发阈值
    keep_last_turns: int = 4  # 保留最近 K 轮原样
    summary_max_chars: int = 600  # 摘要长度


@dataclass
class Compactor:
    policy: CompactionPolicy

    def approx_tokens(self, messages: list[dict]) -> int:
        return sum(len(m.get("content", "")) // 2 for m in messages)

    def needs_compaction(self, messages: list[dict]) -> bool:
        return self.approx_tokens(messages) > self.policy.max_tokens

    def extract_facts(self, messages: list[dict]) -> list[str]:
        """极简事实抽取: 找含 '是'/'叫'/'喜欢' 的句子。"""
        facts = []
        for m in messages:
            c = m.get("content", "")
            for kw in ["是", "叫", "喜欢", "住", "工作", "preference", "is", "lives", "likes"]:
                if kw in c:
                    facts.append(c[:80])
                    break
        return list(dict.fromkeys(facts))[:8]  # 去重, 最多 8 条

    def mock_summarize(self, messages: list[dict]) -> str:
        """用 mock 替代 LLM 摘要。"""
        facts = self.extract_facts(messages)
        summary = "[摘要] " + " | ".join(facts)
        return summary[: self.policy.summary_max_chars]

    def compact(self, messages: list[dict]) -> list[dict]:
        """压缩: 旧消息 -> 摘要, 保留最近 K 轮原样。"""
        if not self.needs_compaction(messages):
            return messages

        # 拆: 老段(待摘要) + 新段(原样)
        split_at = max(0, len(messages) - self.policy.keep_last_turns * 2)
        old, recent = messages[:split_at], messages[split_at:]

        if not old:
            return messages

        summary_text = self.mock_summarize(old)
        compacted = [{"role": "system", "content": summary_text}] + recent
        return compacted


def run_demo() -> None:
    c = Compactor(CompactionPolicy(max_tokens=2000, keep_last_turns=3))

    # 构造一个超长对话
    history = []
    for i in range(30):
        history.append(
            {
                "role": "user",
                "content": f"问题 #{i}: 我叫 Alice, 喜欢科幻片, 住在北京, 是一名工程师",
            }
        )
        history.append({"role": "assistant", "content": f"回答 #{i}: 已记录你的偏好。"})

    print("=== 压缩前 ===")
    print(f"  消息数: {len(history)}  估算 tokens: {c.approx_tokens(history)}")
    print(f"  触发压缩? {c.needs_compaction(history)}")

    compacted = c.compact(history)
    print("\n=== 压缩后 ===")
    print(f"  消息数: {len(compacted)}  估算 tokens: {c.approx_tokens(compacted)}")
    print(f"  节省: {(1 - c.approx_tokens(compacted) / c.approx_tokens(history)):.0%}")
    print("\n--- 第 1 条 (摘要) ---")
    print(compacted[0]["content"])
    print(f"\n--- 后续 {len(compacted) - 1} 条保留原样 ---")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
