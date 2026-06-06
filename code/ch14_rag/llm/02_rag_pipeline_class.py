# ---
# chapter: 14
# topic: RAG 检索生成 Pipeline
# section: 14.2.2 检索+生成阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 02_rag_pipeline_class.py
# expected_runtime: <1s
# expected_output: RAG pipeline demo with mock LLM
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.2-rag-完整架构
# Interview hooks:
#   1. RAG Pipeline 中 retrieve 与 generate 的边界是什么？为什么要分开？
#   2. RAG 场景下 LLM 的 temperature 应该如何设置？为什么？
#   3. 如何设计 RAG Prompt 让模型"基于上下文回答"且不知道时说不知道？

class RAGPipeline:
    """RAG 检索生成 Pipeline - 完整实现（mock LLM 演示版）"""

    def __init__(self, vectorstore, llm_client=None, top_k: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm_client
        self.top_k = top_k
        # RAG Prompt 模板
        self.rag_prompt_template = """基于以下检索到的上下文信息回答问题。
如果上下文中没有相关信息，请明确说明"根据现有资料无法回答"。

上下文：
{context}

---

问题：{question}

请给出准确、简洁的回答。如果涉及数据，请注明来源。"""

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        """
        向量检索：返回 (文档内容, 相似度分数) 列表
        """
        if self.vectorstore is None:
            # Mock 返回
            return [
                ("[文档 1] 年假政策：员工每年享有 15 天带薪年假。", 0.92),
                ("[文档 2] 病假规定：员工请病假需提供医院证明。", 0.78),
            ]
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        return [(doc.page_content, score) for doc, score in results]

    def generate(self, query: str, retrieved_docs: list[tuple[str, float]]) -> str:
        """
        基于检索结果生成回答（mock，不真正调用 LLM）
        """
        context = "\n\n---\n\n".join([
            f"[文档 {i+1}]（相似度：{score:.3f}）\n{content}"
            for i, (content, score) in enumerate(retrieved_docs)
        ])
        prompt = self.rag_prompt_template.format(context=context, question=query)
        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        # Mock 回答
        return f"[Mock Answer] 基于 {len(retrieved_docs)} 个检索文档生成的回答: {query}"

    def query(self, question: str) -> dict:
        """端到端查询"""
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"content": d[:200] + "...", "score": float(s)}
                for d, s in docs
            ],
        }


if __name__ == "__main__":
    rag = RAGPipeline(vectorstore=None, llm_client=None, top_k=5)
    result = rag.query("公司的年假政策是什么？")
    print(f"问题: {result['question']}")
    print(f"回答: {result['answer']}")
    print(f"检索到 {len(result['sources'])} 条来源")
    print("OK")
