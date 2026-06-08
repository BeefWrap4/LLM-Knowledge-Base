# ---
# chapter: 29
# topic: 端到端 Context Pipeline — 输入分类/RAG/压缩/缓存/输出
# section: 29.8 末段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 12_full_context_pipeline.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.8 / §29.10
# Cross-refs:
#   - Ch13 Prompt Engineering (组装层模板)
#   - Ch14 RAG (检索层)
#   - Ch15 Agent (工具层)
#   - Ch18 LangGraph (状态层)
#   - Ch20 LLMOps (成本监控)
#   - Ch25 推理引擎 (Prefix Cache)
#
# Interview hooks:
#   - "如何设计一个 Context Pipeline?"  →  输入/检索/压缩/组装/输出 5 层
#   - "各层关注什么?"                  →  清洗/Top-K/分块摘要/模板+缓存/结构化输出
#   - "Pipeline 如何验证质量?"          →  离线评估集 + 关键事实召回率 + 端到端 task accuracy

from __future__ import annotations

from dataclasses import dataclass, field

# ---- 5 个层 ----


@dataclass
class InputLayer:
    """层 1: 输入 — 清洗 + 意图分类 + 安全过滤。"""

    def run(self, raw_query: str) -> dict:
        q = raw_query.strip()
        intent = "qa" if "?" in q or "？" in q else "chat"
        return {"query": q, "intent": intent, "clean": True}


@dataclass
class RetrievalLayer:
    """层 2: 检索 — RAG + Rerank, Top-K=3。"""

    corpus: list[str] = field(default_factory=list)

    def run(self, query: str, k: int = 3) -> list[str]:
        # 简单关键词命中, mock BM25
        words = set(query.lower().split())
        scored = []
        for c in self.corpus:
            s = sum(1 for w in words if w in c.lower())
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:k]] or ["(无相关文档)"]


@dataclass
class CompressionLayer:
    """层 3: 压缩 — 长文档分块 + 摘要。"""

    chunk_size: int = 120

    def run(self, docs: list[str]) -> list[str]:
        out = []
        for d in docs:
            if len(d) > self.chunk_size:
                # 截断 + 提示
                out.append(d[: self.chunk_size] + "...")
            else:
                out.append(d)
        return out


@dataclass
class AssemblyLayer:
    """层 4: 组装 — Template + Few-shot + RAG + Memory + Cache prefix 标记。"""

    cached_prefix: str = ""  # 稳定部分 (system + few-shot)
    dynamic_part: str = ""  # 动态部分 (RAG + memory + query)

    def run(self, *, intent: str, docs: list[str], memory_hits: list[str], query: str) -> dict:
        self.cached_prefix = "[CACHE-PREFIX]\n你是一个严谨的企业知识助手。\nFew-shot: Q=PE 含义? A=市盈率。"
        rag_block = "\n".join(f"- {d}" for d in docs)
        mem_block = "\n".join(f"- {m}" for m in memory_hits) or "(无 LTM 命中)"
        self.dynamic_part = (
            f"[DYNAMIC]\n意图: {intent}\n相关文档:\n{rag_block}\n长期记忆:\n{mem_block}\n用户问题: {query}"
        )
        full = self.cached_prefix + "\n\n" + self.dynamic_part
        return {
            "prompt": full,
            "cache_prefix_tokens": len(self.cached_prefix) // 2,
            "dynamic_tokens": len(self.dynamic_part) // 2,
        }


@dataclass
class OutputLayer:
    """层 5: 输出 — 结构化 + 后处理。"""

    def run(self, raw: str) -> dict:
        # mock: 强制返回结构化
        return {
            "answer": raw[:200] or "[mock] 这里是回答",
            "citations": ["doc1", "doc2"],
            "confidence": 0.82,
        }


# ---- 编排: 完整 pipeline ----


@dataclass
class ContextPipeline:
    inp: InputLayer = field(default_factory=InputLayer)
    ret: RetrievalLayer = field(default_factory=RetrievalLayer)
    cmp: CompressionLayer = field(default_factory=CompressionLayer)
    asm: AssemblyLayer = field(default_factory=AssemblyLayer)
    out: OutputLayer = field(default_factory=OutputLayer)

    def seed_corpus(self, docs: list[str]) -> None:
        self.ret.corpus = docs

    def run(self, query: str, memory_hits: list[str] | None = None) -> dict:
        memory_hits = memory_hits or []
        # 1
        i = self.inp.run(query)
        # 2
        docs = self.ret.run(i["query"], k=3)
        # 3
        docs = self.cmp.run(docs)
        # 4
        assembled = self.asm.run(intent=i["intent"], docs=docs, memory_hits=memory_hits, query=i["query"])
        # 5
        # mock LLM 输出
        mock_llm_output = f"[mock] 基于 {len(docs)} 文档 + {len(memory_hits)} LTM 命中回答: {query}"
        return self.out.run(mock_llm_output) | {"pipeline": assembled}


def run_demo() -> None:
    pipe = ContextPipeline()
    pipe.seed_corpus(
        [
            "Context Engineering 关注模型每步推理时看到的全部信息 (Anthropic 2026)",
            "Haystack 2.x 提供组件化 pipeline, 支持 RAG + 缓存",
            "LangGraph 用 checkpointer 持久化 agent state, 解决长会话问题",
            "Sub-Agent 模式让每个子任务有独立 context, 避免污染",
        ]
    )

    print("=== 端到端 Context Pipeline 演示 ===\n")
    result = pipe.run(
        query="请介绍 Context Engineering 与 Haystack 的关系?",
        memory_hits=["用户偏好: 关注工程实践", "用户身份: 后端工程师"],
    )
    print("Pipeline 组装:")
    print(f"  cache_prefix_tokens={result['pipeline']['cache_prefix_tokens']}")
    print(f"  dynamic_tokens={result['pipeline']['dynamic_tokens']}")
    print()
    print("最终输出:")
    print(f"  answer: {result['answer']}")
    print(f"  citations: {result['citations']}")
    print(f"  confidence: {result['confidence']}")
    print("\n→ 提示: cache_prefix (system+few-shot) 跨轮命中, dynamic 每轮新鲜。")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
