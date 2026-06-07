# ---
# chapter: 14
# topic: Graph RAG 简化实现
# section: 14.6.1 Graph RAG
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: networkx, sentence-transformers, numpy
# run: python 15_graph_rag.py
# expected_runtime: <1s (mock LLM/embedder)
# expected_output: knowledge graph built and queried
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.6-高级-rag-技术
# Interview hooks:
#   1. Graph RAG 相比普通 RAG 解决了什么本质问题？举例说明多跳推理场景。
#   2. 实体-关系抽取的质量如何保证？错误抽取会如何传播？
#   3. Graph RAG 的构建成本主要由什么决定？什么场景值得用？

# Graph RAG 简化实现
import json

import numpy as np


class GraphRAG:
    """Graph RAG 简化实现（基于 NetworkX）"""

    def __init__(self, llm_client=None, embedder=None):
        self.llm = llm_client
        self.embedder = embedder
        try:
            import networkx as nx
        except ImportError:
            nx = None
        self.graph = nx.Graph() if nx is not None else None
        self._nx = nx

    def extract_entities_relations(self, text: str) -> list[dict]:
        """用 LLM 抽取实体和关系"""
        prompt = f"""从以下文本中提取实体和关系，输出 JSON 格式：
{text}

格式：[{{"subject": "实体1", "relation": "关系", "object": "实体2"}}, ...]
"""
        if self.llm is not None:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        # Mock 三元组抽取
        return [
            {"subject": "Alice", "relation": "reports_to", "object": "Bob"},
            {"subject": "Bob", "relation": "employed_by", "object": "TechCorp"},
        ]

    def build_graph(self, documents: list[str]):
        """从文档构建知识图谱"""
        for doc in documents:
            triples = self.extract_entities_relations(doc)
            for t in triples:
                self.graph.add_edge(
                    t["subject"], t["object"],
                    relation=t["relation"],
                    source=doc[:100],
                )

    def retrieve(self, query: str, max_hops: int = 2) -> list[str]:
        """图检索：从查询实体出发，进行多跳邻居遍历"""
        # 从查询中提取实体（简化：假设查询就是实体名）
        if self.embedder is not None:
            query_embedding = self.embedder.encode(query)
        else:
            query_embedding = np.zeros(16)

        # 找到最匹配的图节点
        best_node = None
        best_score = -1
        for node in self.graph.nodes():
            if self.embedder is not None:
                node_emb = self.embedder.encode(node)
            else:
                # Mock 评分: 完全匹配优先
                node_emb = np.zeros_like(query_embedding)
            score = float(np.dot(query_embedding, node_emb))
            # 简单字符串包含也算
            if query.lower() in str(node).lower():
                score += 1.0
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None:
            return []

        # 多跳邻居遍历
        from collections import deque
        visited = {best_node}
        queue = deque([(best_node, 0)])
        paths = []

        while queue:
            node, hops = queue.popleft()
            if hops > max_hops:
                continue
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                path = f"{node} --[{edge_data['relation']}]--> {neighbor}"
                paths.append(path)
                if neighbor not in visited and hops < max_hops:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))
        return paths


if __name__ == "__main__":
    g = GraphRAG(llm_client=None, embedder=None)
    g.build_graph(["Alice reports to Bob at TechCorp."])
    paths = g.retrieve("Alice", max_hops=2)
    print("图遍历结果:")
    for p in paths:
        print(f"  {p}")
