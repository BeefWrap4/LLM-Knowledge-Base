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


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print("OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)")
import os
from llama_index.core import Settings

# Wave 20: 优先真实 LLM + embedding, 缺 key 降级 mock
USE_REAL_API = os.environ.get("USE_REAL_API") == "1"
if USE_REAL_API:
    # 真实 LLM (默认厂商)
    from shared.chatmodel_factory import make_chat_model
    real_llm = make_chat_model(framework="llama_index")
    if real_llm is not None:
        Settings.llm = real_llm
    else:
        print("[mock] UnifiedClient 无 Key, 降级 mock")
        class _MockLLM:
            def complete(self, prompt, **kwargs):
                return type("R", (), {"text": f"（mock）回答：{prompt[-100:]}"})()
        Settings.llm = _MockLLM()
else:
    # Mock 模式: 模拟 LLM 和 embed
    class _MockEmbed:
        def get_text_embedding(self, text):
            return [float(len(text))] * 8
    class _MockLLM:
        def complete(self, prompt, **kwargs):
            return type("R", (), {"text": f"（mock）回答基于以下知识：{prompt[-200:]}..."})()
    Settings.embed_model = _MockEmbed()
    Settings.llm = _MockLLM()

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