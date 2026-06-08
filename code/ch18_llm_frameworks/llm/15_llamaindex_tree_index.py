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
    from llama_index.core import Document, TreeIndex

    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print(
    "OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)"
)
from llama_index.core import Settings

from shared._error_helper import raise_with_help

# W3-T5: 真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 走 raise_with_help
from shared.chatmodel_factory import make_chat_model

real_llm = make_chat_model(framework="llama_index")
if real_llm is None:
    raise_with_help(
        "需要 LLM_PROVIDER + API Key 来运行此例子.",
        "运行 `make llm-doctor-setup` 配置; 或参考 README §环境配置.",
    )
Settings.llm = real_llm

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

query_engine = tree_index.as_query_engine(response_mode="tree_summarize")
print(
    "TreeIndex 构造完成。索引节点数:",
    len(tree_index.index_struct.all_nodes) if hasattr(tree_index.index_struct, "all_nodes") else "N/A",
)

if __name__ == "__main__":
    print("OK")
