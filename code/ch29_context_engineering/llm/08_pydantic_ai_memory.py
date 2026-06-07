# ---
# chapter: 29
# topic: Pydantic AI 记忆系统 — STM/LTM/Episodic/Procedural 四层
# section: 29.5
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无 (mock)
# run: python 08_pydantic_ai_memory.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.5
# Cross-refs:
#   - Ch15 Agent (memory 集成)
#   - Ch14 RAG (向量检索作为 LTM)
#   - Ch03 OOP (分层抽象)
#
# Interview hooks:
#   - "记忆系统为什么分层?"     →  短/长/情景/程序 各自生命周期/可检索性/更新频率不同
#   - "Pydantic AI 的 MemoryTool?" →  load_recent_messages / vector_search / structured
#   - "记忆系统设计原则?"       →  分层/选择性/可检索/可更新/隐私

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ShortTermMemory:
    """STM: 当前会话的对话历史 (in-context, 速度优先)。"""
    capacity: int = 20
    messages: deque = field(default_factory=lambda: deque(maxlen=20))

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def to_messages(self) -> list[dict]:
        return list(self.messages)


@dataclass
class LongTermMemory:
    """LTM: 用户偏好/事实, 向量检索 (mock 为关键词命中)。"""
    facts: list[dict] = field(default_factory=list)  # [{"text","tags"}]

    def add(self, text: str, tags: list[str]) -> None:
        self.facts.append({"text": text, "tags": tags})

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = set(query.lower().split())
        scored = []
        for f in self.facts:
            s = len(q & set(" ".join(f["tags"]).lower().split())) + (0.5 if any(w in f["text"].lower() for w in q) else 0)
            if s > 0:
                scored.append((s, f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:top_k]]


@dataclass
class EpisodicMemory:
    """Episodic: 过去事件的摘要, 按时间索引。"""
    episodes: list[dict] = field(default_factory=list)  # [{"ts","summary"}]

    def add(self, summary: str) -> None:
        self.episodes.append({"ts": len(self.episodes) + 1, "summary": summary})

    def recent(self, k: int = 3) -> list[dict]:
        return self.episodes[-k:]


@dataclass
class ProceduralMemory:
    """Procedural: 技能/工具使用流程 (instructable, 可被 prompt 注入)。"""
    skills: list[str] = field(default_factory=list)

    def add(self, skill: str) -> None:
        if skill not in self.skills:
            self.skills.append(skill)

    def as_instructions(self) -> str:
        if not self.skills:
            return ""
        return "可用技能: " + ", ".join(self.skills)


class PydanticAIStyleAgent:
    """模拟 pydantic_ai.Agent + memory=[] 用法。"""

    def __init__(self):
        self.stm = ShortTermMemory(capacity=12)
        self.ltm = LongTermMemory()
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()

    def seed(self) -> None:
        self.ltm.add("用户 Alice 喜欢科幻片", ["preference", "movie"])
        self.ltm.add("用户住在北京, 工程师", ["profile", "location", "job"])
        self.ltm.add("偏好风险等级: 中", ["preference", "risk"])
        self.procedural.add("search_web")
        self.procedural.add("query_db")
        self.procedural.add("calc_dcf")

    def ask(self, query: str) -> dict:
        # 1) STM: 加入当前 query
        self.stm.add("user", query)
        # 2) LTM 检索: 把命中事实注入 context
        ltm_hits = self.ltm.search(query, top_k=3)
        # 3) Episodic: 拉最近 2 个事件
        eps = self.episodic.recent(k=2)
        # 4) 组装 context
        context = {
            "stm": self.stm.to_messages(),
            "ltm_hits": ltm_hits,
            "episodic": eps,
            "skills": self.procedural.as_instructions(),
        }
        return context


def run_demo() -> None:
    agent = PydanticAIStyleAgent()
    agent.seed()

    queries = [
        "推荐一部科幻电影?",
        "我住的城市的天气怎么样?",
        "我适合买什么风险等级的基金?",
    ]
    for q in queries:
        print(f"=== User: {q} ===")
        ctx = agent.ask(q)
        print("  STM (最近):", [m["content"][:40] for m in ctx["stm"][-2:]])
        print("  LTM 命中: ", [f["text"] for f in ctx["ltm_hits"]] or "(无)")
        print("  Episodic:  ", [e["summary"] for e in ctx["episodic"]] or "(无)")
        print("  Skills:    ", ctx["skills"] or "(无)")
        print()
        agent.episodic.add(f"用户问: {q}")
        agent.stm.add("assistant", f"[mock] 基于 LTM={len(ctx['ltm_hits'])} 条事实回答: {q}")


if __name__ == "__main__":
    run_demo()
