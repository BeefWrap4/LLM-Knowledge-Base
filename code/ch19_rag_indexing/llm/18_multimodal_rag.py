# ---
# chapter: 21
# topic: 生产级 RAG 系统
# topic_id: rag_indexing.multimodal_rag
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy, scikit-learn
# run: python 18_multimodal_rag.py
# expected_runtime: <1s (mock mode)
# expected_output: multimodal retrieve + generate demo
# ---
# See: ../../../21_生产级RAG系统.md
# Interview hooks:
#   1. 多模态 RAG 相比纯文本 RAG 多哪些挑战（模态对齐、Layout-aware chunking、OCR 质量）？
#   2. CLIP 类的统一向量空间如何实现跨模态检索？
#   3. 哪些场景适合多模态 RAG（产品手册、PPT、技术图表检索）？

# 多模态 RAG 简化实现
import numpy as np


def cosine_similarity(a, b):
    """计算余弦相似度"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


class MultimodalRAG:
    """多模态 RAG：支持文本 + 图像混合检索（mock 演示版）"""

    def __init__(self, text_embedder=None, image_embedder=None, multimodal_llm=None):
        self.text_embedder = text_embedder  # 文本 Embedding 模型
        self.image_embedder = image_embedder  # CLIP/SigLIP 图像编码器
        self.llm = multimodal_llm  # 多模态大模型（如 Qwen-VL）
        self.doc_store: list = []  # 文档存储
        self.image_store: list = []  # 图像存储

    def _mock_embed(self, items, dim=16):
        rng = np.random.default_rng(abs(hash(tuple(items))) % (2**32) if items else 0)
        v = rng.normal(size=(len(items), dim)).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def index_document(self, text_chunks: list[str], images: list = None):
        """索引文档：文本和图像分别编码"""
        images = images or []
        # 文本编码
        if self.text_embedder is not None and hasattr(self.text_embedder, "encode"):
            text_embeddings = self.text_embedder.encode(text_chunks)
        else:
            text_embeddings = self._mock_embed(text_chunks)
        for chunk, emb in zip(text_chunks, text_embeddings):
            self.doc_store.append({"type": "text", "content": chunk, "embedding": emb})

        # 图像编码
        if images:
            if self.image_embedder is not None and hasattr(self.image_embedder, "encode"):
                image_embeddings = self.image_embedder.encode(images)
            else:
                image_embeddings = self._mock_embed([f"img_{i}" for i in range(len(images))])
            for img, emb in zip(images, image_embeddings):
                self.image_store.append({"type": "image", "content": img, "embedding": emb})

    def retrieve(self, query: str, query_image=None, top_k: int = 5):
        """
        多模态检索：支持纯文本查询、纯图像查询、图文查询
        """
        results = []

        # 文本检索
        if self.text_embedder is not None and hasattr(self.text_embedder, "encode"):
            query_text_emb = self.text_embedder.encode([query])
        else:
            query_text_emb = self._mock_embed([query])
        if self.doc_store:
            text_scores = cosine_similarity(
                query_text_emb,
                [d["embedding"] for d in self.doc_store],
            )
            for i, score in enumerate(text_scores[0]):
                results.append(("text", i, float(score)))

        # 图像检索（如果有查询图像）
        if query_image is not None:
            if self.image_embedder is not None and hasattr(self.image_embedder, "encode"):
                query_img_emb = self.image_embedder.encode([query_image])
            else:
                query_img_emb = self._mock_embed(["query_img"])
            if self.image_store:
                img_scores = cosine_similarity(
                    query_img_emb,
                    [d["embedding"] for d in self.image_store],
                )
                for i, score in enumerate(img_scores[0]):
                    results.append(("image", i, float(score)))

        # 按相似度排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def generate(self, query: str, retrieved_items: list) -> str:
        """使用多模态 LLM 生成回答（mock）"""
        # 组装多模态 Prompt（文本 + 图像）
        messages = [{"role": "user", "content": []}]
        for item_type, idx, _score in retrieved_items:
            if item_type == "text":
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": f"[相关文档]\n{self.doc_store[idx]['content']}\n",
                    }
                )
            elif item_type == "image":
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": self.image_store[idx]["content"],
                    }
                )
        messages[0]["content"].append({"type": "text", "text": f"\n问题：{query}"})
        if self.llm is not None and hasattr(self.llm, "chat"):
            return self.llm.chat(messages)
        return f"[Mock multimodal answer] 基于 {len(retrieved_items)} 项多模态检索结果回答: {query}"


if __name__ == "__main__":
    rag = MultimodalRAG(text_embedder=None, image_embedder=None, multimodal_llm=None)
    rag.index_document(
        text_chunks=["RAG 是检索增强生成。", "CLIP 用于图文检索。"],
        images=["fake_img_1", "fake_img_2"],
    )
    results = rag.retrieve("什么是 RAG", query_image=None, top_k=3)
    for item_type, idx, score in results:
        print(f"  type={item_type} idx={idx} score={score:.3f}")
    print("OK")
