# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.7.3 多模态 RAG 系统 - ColPali 风格多向量检索 + 视觉问答
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers, pillow, numpy (真实模式需 PyMuPDF)
# run: python 11_multimodal_rag.py
# expected_runtime: <5s (mock)
# expected_output: 演示 ColBERT 风格的 MaxSim 检索流程
# ---
# See: ../tutorial/21_多模态大模型.md#21-7-3-实战：构建多模态rag系统
# Interview hooks:
#   1. ColPali 为什么比 OCR + 文本检索更适合文档问答？
#   2. ColBERT 风格的 MaxSim 评分如何工作？
#   3. 多模态 RAG 相比纯文本 RAG 在哪些场景下优势明显？

import os
from typing import List, Tuple

import numpy as np


class MultiModalRAG:
    """多模态文档检索与问答系统（ColPali 风格）。"""

    def __init__(
        self,
        retriever_model: str = "vidore/colpali-v1.2",
        generator_model: str = "gpt-4o",
    ):
        self.retriever_model_name = retriever_model
        self.generator_model = generator_model
        # --- 文档存储 ---
        self.doc_embeddings: List[np.ndarray] = []   # List of [num_patches, dim]
        self.doc_images = []                          # 原始文档图像

    def _load_retriever(self):
        """延迟加载真实视觉编码器（仅在非 mock 模式调用）。"""
        import torch
        from transformers import AutoModel, AutoProcessor

        self.retriever = AutoModel.from_pretrained(
            self.retriever_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.retriever_model_name)

    def index_document(self, pdf_path: str, dpi: int = 200):
        """将 PDF 文档转换为图像并索引（真实模式需 PyMuPDF）。"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF (fitz) required for real PDF indexing")

        from PIL import Image
        import torch

        if not hasattr(self, "retriever"):
            self._load_retriever()

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            with torch.no_grad():
                inputs = self.processor(images=img, return_tensors="pt").to(
                    self.retriever.device
                )
                embeddings = self.retriever(**inputs)  # [1, num_patches, dim]
            self.doc_embeddings.append(embeddings[0].cpu().numpy())
            self.doc_images.append(img)
        print(f"已索引 {len(doc)} 页文档")

    def retrieve(
        self, query_emb: np.ndarray, top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """ColBERT 风格的延迟交互：MaxSim 检索。"""
        scores = []
        for page_idx, doc_emb in enumerate(self.doc_embeddings):
            # doc_emb: [num_patches, dim]
            # query_emb: [num_query_tokens, dim]
            q = query_emb
            d = doc_emb
            similarity = q @ d.T                         # [Q, D]
            max_per_query = similarity.max(axis=1)       # [Q]
            score = float(max_per_query.sum())
            scores.append(score)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), scores[idx]) for idx in top_indices]

    def answer(self, query: str, top_k: int = 3) -> str:
        """多模态问答：检索 + 生成（生成部分在 mock 模式下为占位）。"""
        # mock 模式：使用随机 query embedding 演示 MaxSim
        rng = np.random.default_rng(hash(query) % (2**32))
        query_emb = rng.standard_normal((8, 64)).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb, axis=-1, keepdims=True)
        # 至少索引 1 个文档以演示检索
        if not self.doc_embeddings:
            self.doc_embeddings = [rng.standard_normal((49, 64)).astype(np.float32)]
        top = self.retrieve(query_emb, top_k=top_k)
        print(f"Top-{top_k} retrieved pages: {[p for p, _ in top]}")
        return f"[VLM Response based on {len(top)} retrieved images]"


def main():
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    rag = MultiModalRAG()
    if not use_mock:
        # 真实模式需要 PDF 文件
        pdf_path = os.environ.get("CH21_PDF", "report.pdf")
        if os.path.exists(pdf_path):
            rag.index_document(pdf_path)
        else:
            print(f"PDF not found: {pdf_path}, falling back to mock")
    answer = rag.answer("第三季度营收增长了多少？")
    print(answer)


if __name__ == "__main__":
    main()
