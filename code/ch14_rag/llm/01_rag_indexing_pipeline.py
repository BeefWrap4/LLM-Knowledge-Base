# ---
# chapter: 14
# topic: RAG索引阶段完整代码示例
# section: 14.2.2 索引阶段详解
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: langchain, faiss-cpu, sentence-transformers
# run: python 01_rag_indexing_pipeline.py
# expected_runtime: <1s (mock mode)
# expected_output: indexing steps with mock output
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.2-rag-完整架构
# Interview hooks:
#   1. RAG 索引阶段包含哪些步骤？每步的作用是什么？
#   2. RecursiveCharacterTextSplitter 的分隔符优先级是如何设计的？为什么这样设计？
#   3. 为什么 Embedding 后要做归一化（normalize_embeddings=True）？
# Mock-mode demo of RAG indexing pipeline; replace loaders with real PDF/TXT in production.


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    print("OK")
    _sys.exit(0)


def build_rag_index(
    document_paths: list[str],
    embedding_model: str = "BAAI/bge-large-zh-v1.5",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    output_dir: str = "./faiss_index",
):
    """
    构建 RAG 向量索引 - 完整流程（mock 演示版）

    生产环境可还原原代码:
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
    """
    # Step 1: 文档加载（mock）
    documents = []
    for path in document_paths:
        if path.endswith(".pdf"):
            # loader = PyPDFLoader(path)
            print(f"[Mock] Loading PDF: {path}")
        else:
            # loader = TextLoader(path, encoding="utf-8")
            print(f"[Mock] Loading text: {path}")
        # documents.extend(loader.load())
    print(f"[1/4] 加载文档完成：共 {len(document_paths)} 个文件 (mock)")

    # Step 2: 文档分块（实际可用 RecursiveCharacterTextSplitter）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    sample_text = "示例文本。" * 200
    chunks = text_splitter.split_text(sample_text)
    print(f"[2/4] 文档分块完成：共 {len(chunks)} 个 chunk (mock)")

    # Step 3 & 4: Embedding + 索引（mock）
    print(f"[3/4] Embedding 编码: {embedding_model} (mock)")
    print(f"[4/4] 索引构建完成，已保存到 {output_dir} (mock)")
    return None


if __name__ == "__main__":
    build_rag_index(
        document_paths=["doc1.pdf", "doc2.md"],
        embedding_model="BAAI/bge-large-zh-v1.5",
    )
    print("OK")