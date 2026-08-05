# ---
# chapter: 26
# topic: Agent 记忆与个性化
# topic_id: context_engineering.pydantic_ai_memory
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 08_pydantic_ai_memory.py
# expected_runtime: <1s
# ---
#
# See: ../../../26_Agent记忆与个性化.md
# Cross-refs:
#   - Ch15 Agent (memory 集成)
#   - Ch14 RAG (向量检索作为 LTM)
#   - Ch03 OOP (分层抽象)
#
# Important:
#   这是纯本地、框架无关的架构示例，不是 Pydantic AI API，也没有调用模型。
#   Pydantic AI 的当前消息历史 API 请见:
#   https://ai.pydantic.dev/message-history/

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ShortTermMemory:
    """STM：当前会话的有限消息窗口。"""

    capacity: int = 20
    messages: deque[dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity 必须大于 0")
        self.messages = deque(maxlen=self.capacity)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def to_messages(self) -> list[dict[str, str]]:
        return list(self.messages)


@dataclass
class LocalLongTermMemory:
    """LTM：本地关键词检索示意；生产环境可替换为数据库或向量检索。"""

    facts: list[dict[str, object]] = field(default_factory=list)

    def add(self, text: str, keywords: list[str]) -> None:
        self.facts.append({"text": text, "keywords": [keyword.lower() for keyword in keywords]})

    def search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        normalized_query = query.lower()
        scored: list[tuple[int, dict[str, object]]] = []
        for fact in self.facts:
            keywords = fact["keywords"]
            assert isinstance(keywords, list)
            score = sum(keyword in normalized_query for keyword in keywords)
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]


@dataclass
class EpisodicMemory:
    """Episodic：过去事件摘要；真实系统还需时间、来源和删除策略。"""

    episodes: list[dict[str, object]] = field(default_factory=list)

    def add(self, summary: str) -> None:
        self.episodes.append({"sequence": len(self.episodes) + 1, "summary": summary})

    def recent(self, limit: int = 3) -> list[dict[str, object]]:
        return self.episodes[-limit:]


@dataclass
class ProceduralMemory:
    """Procedural：可注入上下文的技能说明，不等于真实工具权限。"""

    skills: list[str] = field(default_factory=list)

    def add(self, skill: str) -> None:
        if skill not in self.skills:
            self.skills.append(skill)

    def as_instructions(self) -> str:
        return "可用技能: " + ", ".join(self.skills) if self.skills else ""


class LocalLayeredMemory:
    """组合四层本地存储并产出待注入模型的候选 context。"""

    def __init__(self, short_term_capacity: int = 12):
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LocalLongTermMemory()
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()

    def seed_demo_data(self) -> None:
        self.long_term.add("用户 Alice 喜欢科幻片", ["科幻", "电影"])
        self.long_term.add("用户住在北京，是工程师", ["北京", "城市", "天气"])
        self.long_term.add("用户偏好中等风险", ["风险", "基金"])
        self.procedural.add("search_web")
        self.procedural.add("query_db")

    def assemble_context(self, query: str) -> dict[str, object]:
        self.short_term.add("user", query)
        return {
            "short_term": self.short_term.to_messages(),
            "long_term_hits": self.long_term.search(query, top_k=3),
            "recent_episodes": self.episodic.recent(limit=2),
            "skill_instructions": self.procedural.as_instructions(),
        }


def run_demo() -> None:
    memory = LocalLayeredMemory(short_term_capacity=4)
    memory.seed_demo_data()

    print("=== 框架无关的本地分层记忆示例（未调用模型） ===\n")
    for query in ("推荐一部科幻电影", "北京的天气怎么样", "我适合买什么风险等级的基金"):
        context = memory.assemble_context(query)
        long_term_hits = context["long_term_hits"]
        recent_episodes = context["recent_episodes"]
        assert isinstance(long_term_hits, list)
        assert isinstance(recent_episodes, list)

        print(f"User: {query}")
        print("  LTM 命中:", [fact["text"] for fact in long_term_hits] or "(无)")
        print("  Episodic:", [episode["summary"] for episode in recent_episodes] or "(无)")
        print("  Skills:", context["skill_instructions"] or "(无)")
        memory.episodic.add(f"用户问过：{query}")
        memory.short_term.add("assistant", "[offline placeholder]")
        print()

    print("生产化还需：可信来源、权限隔离、加密、过期/删除、冲突更新与检索评测。")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
