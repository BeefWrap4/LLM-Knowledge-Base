# ---
# chapter: 14
# topic: 递归字符分块
# section: 14.3.2 递归字符分块（最常用）
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain
# run: python 03_recursive_text_splitter.py
# expected_runtime: <1s
# expected_output: chunks count and samples
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.3-文档处理与分块策略
# Interview hooks:
#   1. 为什么 RecursiveCharacterTextSplitter 是 RAG 中最常用的分块器？
#   2. chunk_size 和 chunk_overlap 如何选择？有什么经验值？
#   3. 分隔符顺序（"\n\n" → "\n" → "。"）的意义是什么？



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


# 递归字符分块：按优先级尝试不同分隔符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # 每个 chunk 的目标大小
    chunk_overlap=128,     # 相邻 chunk 的重叠量（关键！保持上下文连贯）
    length_function=len,   # 长度计算函数
    # 分隔符优先级：先按大段落分，不行再按句子，最后按字符
    separators=[
        "\n\n",      # 优先：段落分隔
        "\n",        # 其次：换行
        "。", "！", "？",  # 再其次：句子结束符
        "；",        # 分号
        " ",         # 空格
        ""           # 最后：任意字符
    ],
    is_separator_regex=False,
)


if __name__ == "__main__":
    long_document_text = (
        "## RAG 概述\n\nRAG 是检索增强生成的缩写。\n\n"
        "## 工作原理\n\nRAG 通过检索外部知识库，将相关文档与用户查询一起输入大模型。\n\n"
        "## 优势\n\nRAG 可以缓解幻觉问题，让模型回答有据可循。"
    ) * 5
    chunks = text_splitter.split_text(long_document_text)
    print(f"分块完成: {len(chunks)} 个 chunk")
    for i, c in enumerate(chunks[:3]):
        print(f"--- chunk {i} (len={len(c)}) ---\n{c[:80]}...")
    print("OK")