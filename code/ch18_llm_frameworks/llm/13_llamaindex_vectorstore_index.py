import os
import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.2 VectorStoreIndex
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index, llama-index-embeddings-openai, llama-index-llms-openai
# run: python 13_llamaindex_vectorstore_index.py
# expected_runtime: <1s
# expected_output: query response
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.2
# Interview hooks:
#   1. VectorStoreIndex 的底层是如何将文本转为向量的？涉及哪些组件？
#   2. 为什么语义检索相比关键词检索有更好的"召回率"？
# VectorStoreIndex：向量索引（最常用）

if os.environ.get("LLM_MOCK") != "0":
    print("[offline] VectorStoreIndex: 3 documents -> top-k retrieval -> synthesized answer")
    print("回答: 7 天内可无理由退换；7 至 15 天内质量问题可换货。")
    print("OK")
    raise SystemExit(0)

# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex

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

# Wave 29: 同时配置真实 embedding (本地 bge-small-zh, 避免 OpenAI API 调用)
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

# 加载文档（这里使用内存构造的 Document 列表，不依赖目录）
documents = [
    Document(text="公司退换货政策：自购买之日起7天内可无理由退换货，商品需保持原包装完整。"),
    Document(text="公司退换货政策：超过7天但仍在15天内，如商品有质量问题可申请换货。"),
    Document(text="公司的年假政策：工作满1年员工享受5天年假，满5年员工享受10天年假。"),
]

# 构建向量索引（自动分块+向量化+存储）
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("公司的退换货政策是什么？")
print(response)

if __name__ == "__main__":
    print("OK")
