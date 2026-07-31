# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.2 SummaryIndex
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 14_llamaindex_summary_index.py
# expected_runtime: <1s
# expected_output: summary response
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.2
# Interview hooks:
#   1. SummaryIndex 适合什么场景？tree_summarize 模式如何工作？
#   2. SummaryIndex 与 VectorStoreIndex 在性能与效果上如何权衡？

import os

if os.environ.get("LLM_MOCK") != "0":
    print("[offline] SummaryIndex tree_summarize: LangChain、LangGraph 与 LlamaIndex 各司其职。")
    print("OK")
    raise SystemExit(0)

# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from llama_index.core import Document, SummaryIndex

    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print(
    "OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)"
)
import sys as _sys_path_setup
from pathlib import Path as _Path_setup

from llama_index.core import Settings

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# W3-T5: 真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 走 raise_with_help
from shared._error_helper import raise_with_help
from shared.chatmodel_factory import make_chat_model

real_llm = make_chat_model(framework="llama_index")
if real_llm is None:
    raise_with_help(
        "需要 LLM_PROVIDER + API Key 来运行此例子.",
        "运行 `make llm-doctor-setup` 配置; 或参考 README §环境配置.",
    )
Settings.llm = real_llm

# Wave 29: 真实 embedding (本地 bge)
from pathlib import Path as _P

_bge_path = _P(__file__).resolve().parent.parent.parent / "models" / "bge-small-zh-v1.5"
if not (_bge_path.exists() and (_bge_path / "config.json").exists()):
    raise_with_help(
        f"需要本地 bge 模型权重: {_bge_path}",
        "运行 `make download-models-default` 下载 (或 `setup_local.sh`).",
    )
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=str(_bge_path))
    print(f"[embedding] 使用本地 bge: {_bge_path}")
except ImportError as _e:
    raise_with_help(
        f"需要 llama_index.embeddings.huggingface 才能用本地 bge: {_e}",
        "运行 `pip install llama-index-embeddings-huggingface` (或 `make install-llm`).",
    )

documents = [
    Document(text="文档1: 关于 LangChain 的核心组件与 LCEL 编程范式。"),
    Document(text="文档2: LangGraph 提供状态图与多 Agent 协作能力。"),
    Document(text="文档3: LlamaIndex 专注数据索引与 RAG 检索增强生成。"),
]

# SummaryIndex 适合需要对整个文档集合进行概括的场景
summary_index = SummaryIndex.from_documents(documents)

query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"  # 递归摘要模式
)
response = query_engine.query("总结所有文档的核心观点")
print(response)

if __name__ == "__main__":
    print("OK")
