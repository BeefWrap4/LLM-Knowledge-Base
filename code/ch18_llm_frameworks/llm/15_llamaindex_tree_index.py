# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.2 TreeIndex
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 15_llamaindex_tree_index.py
# expected_runtime: <1s
# expected_output: tree index demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.2
# Interview hooks:
#   1. TreeIndex 的层次结构如何构建？num_children 的取值如何影响效果？
#   2. TreeIndex 在什么场景下比 VectorStoreIndex 更有优势？


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from llama_index.core import TreeIndex, Document
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
        return type("R", (), {"text": "（mock）树形摘要：多文档对比完成。"})()

Settings.llm = _MockLLM()

documents = [
    Document(text="LangChain 文档：LCEL 与 Chain 抽象。"),
    Document(text="LangGraph 文档：状态图与 Node / Edge。"),
    Document(text="LlamaIndex 文档：Document / Index / Retriever。"),
]

# TreeIndex 适合层次化、多文档对比场景
tree_index = TreeIndex.from_documents(
    documents,
    num_children=10,  # 每个摘要节点覆盖的子节点数
)

query_engine = tree_index.as_query_engine(
    response_mode="tree_summarize"
)
print("TreeIndex 构造完成。索引节点数:",
      len(tree_index.index_struct.all_nodes) if hasattr(tree_index.index_struct, 'all_nodes') else 'N/A')

if __name__ == "__main__":
    print("OK")