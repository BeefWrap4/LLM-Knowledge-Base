# ---
# chapter: 29
# topic: Context Engineering 四大组成 (Instructions/Knowledge/Tools/State)
# section: 29.2
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无 (纯 stdlib + mock)
# run: python 01_context_engineering_intro.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.2
# Cross-refs:
#   - Ch13 Prompt Engineering (Instructions 维度)
#   - Ch14 RAG (Knowledge 维度)
#   - Ch15 Agent (Tools 维度)
#   - Ch18 LangGraph (State 维度)
#
# Interview hooks:
#   - "Context Engineering 与 Prompt Engineering 区别?"  →  Prompt 是单次指令; Context 是模型每步所见全部信息 (4 维)
#   - "Context Engineering 关注什么?"                    →  放什么/何时放/怎么放/何时不放
#   - "Context = ?"                                      →  Prompt + History + Tools + RAG + Memory + State

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instructions:
    """维度 1: 指令 — 静态、稳定的部分。"""

    system_prompt: str
    few_shot: list[dict] = field(default_factory=list)
    output_schema: str = "free-form"

    def token_estimate(self) -> int:
        # 粗略估算: 1 token ≈ 2 中文字 或 0.75 英文词
        return len(self.system_prompt) // 2 + sum(len(ex.get("content", "")) for ex in self.few_shot) // 2


@dataclass
class Knowledge:
    """维度 2: 知识 — 检索/外部数据, 动态注入。"""

    rag_chunks: list[str] = field(default_factory=list)
    db_results: list[Any] = field(default_factory=list)
    web_results: list[str] = field(default_factory=list)

    def token_estimate(self) -> int:
        return sum(len(c) // 2 for c in self.rag_chunks) + sum(len(c) // 2 for c in self.web_results)


@dataclass
class Tools:
    """维度 3: 工具 — 当前可用的能力与最近一次执行状态。"""

    available: list[str] = field(default_factory=list)  # tool names
    schemas: dict = field(default_factory=dict)  # name -> JSON-schema
    last_outputs: dict = field(default_factory=dict)  # name -> last result

    def token_estimate(self) -> int:
        return sum(len(s) // 4 for s in self.schemas.values()) + sum(
            len(str(v)) // 2 for v in self.last_outputs.values()
        )


@dataclass
class State:
    """维度 4: 状态 — 对话历史与结构化状态。"""

    history: list[dict] = field(default_factory=list)
    long_term: list[str] = field(default_factory=list)
    structured: dict = field(default_factory=dict)

    def token_estimate(self) -> int:
        return sum(len(m.get("content", "")) // 2 for m in self.history) + sum(
            len(s) // 2 for s in self.long_term
        )


@dataclass
class Context:
    """Context = 4 维组合, 即模型每步推理时看到的全部信息。"""

    instructions: Instructions
    knowledge: Knowledge
    tools: Tools
    state: State

    def total_tokens(self) -> int:
        return (
            self.instructions.token_estimate()
            + self.knowledge.token_estimate()
            + self.tools.token_estimate()
            + self.state.token_estimate()
        )

    def breakdown(self) -> dict[str, int]:
        return {
            "instructions": self.instructions.token_estimate(),
            "knowledge": self.knowledge.token_estimate(),
            "tools": self.tools.token_estimate(),
            "state": self.state.token_estimate(),
        }

    def render(self) -> str:
        """组装最终送给 LLM 的 context 字符串。"""
        parts = [
            f"[SYSTEM]\n{self.instructions.system_prompt}",
            f"[FEW-SHOT x{len(self.instructions.few_shot)}]\n"
            + "\n".join(f"  Q: {ex.get('q', '')}  A: {ex.get('a', '')}" for ex in self.instructions.few_shot),
            f"[TOOLS: {', '.join(self.tools.available) or 'none'}]\n"
            + "\n".join(f"  schema({n})={s[:80]}" for n, s in self.tools.schemas.items()),
            f"[RAG chunks x{len(self.knowledge.rag_chunks)}]\n"
            + "\n".join(f"  - {c[:80]}" for c in self.knowledge.rag_chunks),
            f"[STATE history x{len(self.state.history)}, LTM x{len(self.state.long_term)}]\n"
            + "\n".join(
                f"  {m.get('role', '?')}: {m.get('content', '')[:60]}" for m in self.state.history[-3:]
            ),
        ]
        return "\n\n".join(parts)


def run_demo() -> None:
    """演示: 构造一个完整的 Context 并显示其组成与 token 估算。"""
    ctx = Context(
        instructions=Instructions(
            system_prompt="你是一个严谨的金融分析师, 引用数据时必须标注文献来源。",
            few_shot=[
                {"q": "PE 是什么?", "a": "市盈率 (Price/Earnings) = 股价 / 每股盈利。"},
                {"q": "ROE 含义?", "a": "净资产收益率, 衡量股东回报率。"},
            ],
            output_schema="json",
        ),
        knowledge=Knowledge(
            rag_chunks=[
                "2026 Q1 沪深 300 净利润同比 +4.2%",
                "央行 2026/04 降准 25bp 释放流动性约 5000 亿",
            ],
            web_results=["高盛报告: 中国权益资产超配"],
        ),
        tools=Tools(
            available=["search_web", "query_db", "calc"],
            schemas={"search_web": '{"query": "string"}', "query_db": '{"sql": "string"}'},
            last_outputs={"search_web": "ok 3 hits"},
        ),
        state=State(
            history=[
                {"role": "user", "content": "分析一下当前 A 股估值"},
                {"role": "assistant", "content": "请提供时间窗口与板块偏好"},
            ],
            long_term=["用户偏好: 关注科技/医药板块", "用户风险偏好: 中等"],
            structured={"turn_id": 3, "intent": "valuation_analysis"},
        ),
    )

    print("=== Context 四大组成 ===")
    for dim, n in ctx.breakdown().items():
        print(f"  {dim:<14s} ~{n:>5d} tokens")
    print(f"  {'TOTAL':<14s} ~{ctx.total_tokens():>5d} tokens")
    print("\n=== 组装后的 context (截断) ===\n")
    print(ctx.render()[:600] + "\n...")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
