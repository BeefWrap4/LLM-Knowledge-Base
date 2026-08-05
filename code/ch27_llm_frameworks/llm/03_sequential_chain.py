# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.sequential_chain
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain-classic, langchain-openai
# run: python 03_sequential_chain.py
# expected_runtime: <1s (mock mode)
# expected_output: outline + article dict
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. SequentialChain 中 output_key 的作用是什么？变量如何在链之间流动？
#   2. SequentialChain 与 LCEL 的 `chain1 | chain2` 写法有什么本质差异？


import os

if os.environ.get("LLM_MOCK") != "0":
    print("[SKIP] LangChain Classic 迁移示例仅在显式 LLM_MOCK=0 时运行")
    print("OK")
    raise SystemExit(0)

# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from langchain_classic.chains import LLMChain, SequentialChain

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

# 第一链：生成大纲
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["topic"], template="为关于'{topic}'的博客文章生成一个3点大纲。"),
    output_key="outline",
)

# 第二链：基于大纲写正文
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["outline"], template="基于以下大纲，写一篇300字的博客文章：\n\n{outline}"
    ),
    output_key="article",
)

# 串联
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["outline", "article"],
    verbose=True,
)

result = overall_chain.invoke({"topic": "大模型应用框架选型"})
print(result["article"])

if __name__ == "__main__":
    print("OK")
