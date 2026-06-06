# ---
# chapter: 14
# topic: Agentic RAG
# section: 14.6.2 Agentic RAG
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: (none — uses stdlib + langchain optional)
# run: python 16_agentic_rag.py
# expected_runtime: <1s (mock mode)
# expected_output: agentic RAG plan + retrieve + self-check loop demo
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.6-高级-rag-技术
# Interview hooks:
#   1. Agentic RAG 与普通 RAG 的本质架构差异是什么？
#   2. self_check 阶段如何检测幻觉？实现成本如何？
#   3. max_iterations 应该如何设置？过大/过小有什么副作用？

# Agentic RAG 核心实现
import json


class AgenticRAG:
    """
    Agentic RAG：Agent 驱动的自适应检索（mock LLM 演示版）

    核心特点：
    1. 路由决策：根据查询类型选择检索策略
    2. 多步检索：信息不足时自动补充检索
    3. 自我校验：生成后校验答案与检索结果的一致性
    """

    def __init__(self, vectorstore=None, llm_client=None, tools: dict = None):
        self.vectorstore = vectorstore
        self.llm = llm_client
        self.tools = tools or {}

    def plan(self, query: str) -> dict:
        """规划：决定检索策略"""
        prompt = f"""分析以下查询，选择最佳检索策略。

可用工具：
- vector_search: 向量检索私有知识库
- web_search: 互联网搜索最新信息
- calculator: 数学计算
- multi_query: 将复杂查询分解为多个子查询

用户查询：{query}

请输出 JSON 格式："""
        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        # Mock: 简单规则路由
        if "多少" in query or "计算" in query:
            strategy = "multi_query"
        else:
            strategy = "vector_search"
        return {
            "strategy": strategy,
            "reasoning": "Mock routing decision",
            "sub_queries": [query],
            "tools": [],
        }

    def retrieve(self, strategy: dict, query: str) -> list[str]:
        """执行检索"""
        documents = []
        s = strategy.get("strategy", "vector_search")

        if s == "vector_search":
            if self.vectorstore is not None and hasattr(self.vectorstore, "similarity_search"):
                docs = self.vectorstore.similarity_search(query, k=5)
                documents = [d.page_content for d in docs]
            else:
                documents = [f"[Mock doc] {query} 相关内容"]

        elif s == "multi_query":
            for sq in strategy.get("sub_queries", [query]):
                if self.vectorstore is not None and hasattr(self.vectorstore, "similarity_search"):
                    docs = self.vectorstore.similarity_search(sq, k=3)
                    documents.extend([d.page_content for d in docs])
                else:
                    documents.append(f"[Mock sub-doc] {sq}")

        elif s == "hybrid":
            documents.append(f"[Mock vector] {query}")
            for tool_name in strategy.get("tools", []):
                if tool_name in self.tools:
                    result = self.tools[tool_name](query)
                    documents.append(f"[{tool_name}结果]: {result}")
        return documents

    def self_check(self, query: str, answer: str, sources: list[str]) -> tuple[bool, str]:
        """自我校验：检查答案是否与检索结果一致"""
        prompt = f"""校验以下回答是否与提供的来源信息一致。

来源信息：
{chr(10).join(sources[:3])}

回答：{answer}

如果回答中的事实都能在来源中找到依据，输出 "CONSISTENT"。
如果回答中包含来源中没有的信息（可能是幻觉），输出 "INCONSISTENT: 具体原因"。
"""
        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = response.choices[0].message.content
            return "CONSISTENT" in result, result
        # Mock: 简单一致性 = 答案中包含至少一个 source 关键词
        if not sources:
            return False, "INCONSISTENT: no sources"
        first_source_words = set(sources[0].split())
        answer_words = set(answer.split())
        if first_source_words & answer_words:
            return True, "CONSISTENT (mock)"
        return False, "INCONSISTENT: no overlap (mock)"

    def query(self, question: str, max_iterations: int = 3) -> dict:
        """端到端查询（含规划+检索+校验循环）"""
        all_sources: list[str] = []
        answer = ""
        check_result = ""
        for i in range(max_iterations):
            # 规划
            strategy = self.plan(question)
            # 检索
            sources = self.retrieve(strategy, question)
            all_sources.extend(sources)
            # 生成
            context = "\n\n---\n\n".join(all_sources[:8])  # 限制上下文长度
            prompt = f"""基于以下信息回答问题：
{context}

问题：{question}

请给出准确、简洁的回答。"""
            if self.llm is not None:
                response = self.llm.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
            else:
                answer = f"[Mock answer] {context[:80]}"

            # 自我校验
            is_consistent, check_result = self.self_check(question, answer, all_sources)
            if is_consistent:
                return {
                    "answer": answer,
                    "sources": all_sources,
                    "iterations": i + 1,
                    "check": "passed",
                }
            # 不一致时，将校验反馈加入上下文，下一轮重新规划
            question += f"\n\n[注意：上次回答校验未通过，原因：{check_result}，请修正]"

        return {
            "answer": answer,
            "sources": all_sources,
            "iterations": max_iterations,
            "check": "max iterations reached",
        }


if __name__ == "__main__":
    rag = AgenticRAG(vectorstore=None, llm_client=None, tools={})
    out = rag.query("RAG 是什么？", max_iterations=2)
    print(f"answer: {out['answer']}")
    print(f"iterations: {out['iterations']}")
    print(f"check: {out['check']}")
    print("OK")
