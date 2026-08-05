import sys as _sys_path_setup
from pathlib import Path as _Path_setup

_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.llmchain_basic
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain-classic, langchain-openai
# run: python 02_llmchain_basic.py
# expected_runtime: <1s (mock mode)
# expected_output: ad copy string
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. LLMChain 的 input_variables 有什么作用？模板插值的机制是什么？
#   2. 为什么 LLMChain 在 LCEL 出现后被认为"过时"？


import os

if os.environ.get("LLM_MOCK") != "0":
    print("[SKIP] LangChain Classic 迁移示例仅在显式 LLM_MOCK=0 时运行")
    print("OK")
    raise SystemExit(0)

# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from langchain_classic.chains import LLMChain

    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
print(
    "OK  [hint] pip install -r requirements-llm.txt 后此例子会自动使用真实 LLM (UnifiedClient/chatmodel_factory)"
)
from langchain_core.prompts import PromptTemplate

# Wave 30+: 真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 时 raise
from shared.chatmodel_factory import make_chat_model

llm = make_chat_model()  # 默认厂商 (deepseek)

prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="为{product}写一段面向{audience}的广告文案，50字以内。",
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"product": "智能手表", "audience": "运动爱好者"})
print(result["text"])

if __name__ == "__main__":
    print("OK")
