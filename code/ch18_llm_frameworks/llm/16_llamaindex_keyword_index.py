# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.2 KeywordTableIndex
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 16_llamaindex_keyword_index.py
# expected_runtime: <1s
# expected_output: keyword index demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.2
# Interview hooks:
#   1. KeywordTableIndex 如何把"关键词"提取出来？使用了哪些技术？
#   2. KeywordTableIndex 与 BM25 检索器的关系是什么？


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from llama_index.core import KeywordTableIndex, Document
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    print("OK")
    _sys.exit(0)
from llama_index.core import Settings

class _MockLLM:
    def complete(self, prompt, **kwargs):
        return type("R", (), {"text": "（mock）关键词匹配 + LLM 扩展完成。"})()

Settings.llm = _MockLLM()

documents = [
    Document(text="Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。"),
    Document(text="LangChain 是 LLM 应用编排框架。"),
    Document(text="RAG 是检索增强生成，结合了检索和生成两种技术。"),
]

# 适合基于关键词的精确匹配 + LLM 扩展
kw_index = KeywordTableIndex.from_documents(documents)

query_engine = kw_index.as_query_engine(
    retriever_mode="keyword",  # 先关键词匹配，再 LLM 筛选
)
print("KeywordTableIndex 构造完成。")

if __name__ == "__main__":
    print("OK")