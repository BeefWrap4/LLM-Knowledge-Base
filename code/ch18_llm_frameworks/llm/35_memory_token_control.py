# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 面试真题 18-7：Memory Token 消耗控制
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 35_memory_token_control.py
# expected_runtime: <1s (mock mode) / 5-30s (real API)
# expected_output: memory config dump
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.7 (面试真题精选)
# Interview hooks:
#   1. 在生产环境中如何选择合适的 Memory 策略？需要考虑哪些权衡？
#   2. 为什么推荐用便宜的模型（如 gpt-4o-mini）做摘要？
# 推荐配置：Summary + Buffer 混合策略
import sys as _sys_path_setup
from pathlib import Path as _Path_setup
_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

# W3-T5: 真实 LLM (UnifiedClient + chatmodel_factory), 缺 key 走 raise_with_help
from shared.chatmodel_factory import make_chat_model
from shared._error_helper import raise_with_help
llm = make_chat_model()  # 默认厂商 (cheap mini model)
if llm is None:
    raise_with_help(
        "需要 LLM_PROVIDER + API Key 来运行此例子.",
        "运行 `make llm-doctor-setup` 配置; 或参考 README §环境配置.",
    )

memory_config = {
    "type": "ConversationSummaryBufferMemory",
    "llm": llm,  # 摘要用便宜模型
    "max_token_limit": 2000,    # 总预算
    "return_messages": True,
}

print("=== Memory 配置 ===")
for k, v in memory_config.items():
    if k == "llm":
        # ChatOpenAI exposes .model_name
        print(f"  {k}: <{getattr(v, 'model_name', getattr(v, 'model', 'unknown'))}>")
    else:
        print(f"  {k}: {v}")

print("\n=== 5 种 Memory 策略对比 ===")
strategies = {
    "ConversationBufferMemory": "完整存储，Token 线性增长",
    "ConversationBufferWindowMemory": "保留最近 K 轮，固定 Token",
    "ConversationSummaryMemory": "LLM 压缩历史为摘要",
    "ConversationSummaryBufferMemory": "摘要 + 最近 K 轮（推荐）",
    "ConversationTokenBufferMemory": "按 Token 数硬截断",
    "VectorStoreRetrieverMemory": "向量化按需检索",
}
for k, v in strategies.items():
    print(f"  - {k}: {v}")

if __name__ == "__main__":
    print("OK")
