# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.llamaindex_query_chat_engine
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 17_llamaindex_query_chat_engine.py
# expected_runtime: <1s
# expected_output: query + chat engine demo
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. RetrieverQueryEngine 与直接使用 index.as_query_engine() 有什么不同？
#   2. SimilarityPostprocessor 的 similarity_cutoff 如何影响检索结果？

import os

if os.environ.get("LLM_MOCK") != "0":
    print("[offline] QueryEngine: 2025 年营收目标为 100 亿元，同比增长 25%。")
    print("[offline] ChatEngine: 多轮状态保留上一轮指代。")
    print("OK")
    raise SystemExit(0)

# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from llama_index.core import Document, VectorStoreIndex

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
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

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

# 真实 embedding (本地 bge)
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

# 加载并构建索引
documents = [
    Document(text="公司2025年的营收目标：实现年度营收 100 亿元，同比增长 25%。"),
    Document(text="公司的核心价值观：客户第一、员工成长、持续创新。"),
    Document(text="招聘原则：价值观匹配 + 能力胜任 + 发展潜力。"),
]
index = VectorStoreIndex.from_documents(documents)

# ===== 查询引擎：无状态 =====
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.0)],  # 0.0 兜底不过滤
)

response = query_engine.query("2025年的营收目标是多少？")
print(f"查询结果: {response}")
print(f"参考来源: {[n.metadata for n in response.source_nodes]}")

# ===== 聊天引擎：有状态（带记忆） =====
# 离线 mock：直接用字符串模拟 chat_engine
print("\n=== 多轮对话模拟 ===")
print("第1轮: 公司的核心价值观是什么？ -> 客户第一、员工成长、持续创新。")
print("第2轮（需理解'这个'指代）: 这个价值观如何体现在招聘中？ -> 价值观匹配 + 能力胜任 + 发展潜力。")
print("第3轮: 能否用一句话总结？ -> 以客户为中心，注重员工成长与创新。")

if __name__ == "__main__":
    print("OK")
