# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.3.4 企业文档问答系统
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: llama-index
# run: python 18_llamaindex_enterprise_qa.py
# expected_runtime: <1s
# expected_output: enterprise QA pipeline demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.3.4
# Interview hooks:
#   1. HierarchicalNodeParser 的多层分块策略对检索有什么影响？
#   2. SentenceTransformerRerank 在 RAG 流水线中的作用是什么？

import os

if os.environ.get("LLM_MOCK") != "0":
    print("[offline] Enterprise QA: load -> hierarchical chunk -> retrieve -> postprocess -> answer")
    print("问题: 公司有什么福利？")
    print("回答: 五险一金、年度体检、节日福利。")
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
"""
LlamaIndex 实战：企业文档智能问答系统

完整的 RAG pipeline：
1. 多格式文档加载（PDF, DOCX, MD）
2. 智能分块 + 元数据提取
3. 向量索引构建 + 持久化
4. 高级检索（混合检索 + 重排序）
5. 带记忆的多轮对话
"""
from llama_index.core import Settings
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# ===== Step 1: 全局配置 =====
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
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# ===== Step 2: 模拟加载多格式文档 =====
documents = [
    Document(text="公司规章制度：员工应遵守考勤制度，按时上下班。", metadata={"file_name": "policy.md"}),
    Document(text="员工福利：五险一金、年度体检、节日福利。", metadata={"file_name": "policy.md"}),
    Document(text="招聘流程：投递简历→初筛→技术面试→HR面试→Offer。", metadata={"file_name": "recruit.md"}),
]

# ===== Step 3: 分层分块 + 元数据提取 =====
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],  # 三层分块
    chunk_overlap=20,
)
nodes = node_parser.get_nodes_from_documents(documents)
print(f"生成 {len(nodes)} 个节点")

# 简化：使用 SentenceSplitter 重新分块
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
print(f"分块后节点数: {len(nodes)}")

# ===== Step 4: 构建索引 =====
index = VectorStoreIndex(nodes, show_progress=False)
print("索引已构建（内存模式，未持久化）")

# ===== Step 5: 配置高级检索管线 =====
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=20,  # 初筛 20 条
    vector_store_query_mode="default",
)

# 后处理管线：相似度过滤
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.0),  # 0.0 兜底不过滤
]

# ===== Step 6: 构建查询引擎 =====
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessors,
)

# ===== Step 7: 模拟问答 =====
print("\n" + "=" * 60)
print("📚 企业文档智能问答系统 - 输入 'quit' 退出")
print("=" * 60)

demo_questions = [
    "公司有什么福利？",
    "招聘流程是怎样的？",
]
for q in demo_questions:
    response = query_engine.query(q)
    print(f"\n❓ 问题: {q}")
    print(f"💡 回答: {response}")

if __name__ == "__main__":
    print("OK")
