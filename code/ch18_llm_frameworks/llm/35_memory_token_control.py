# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 面试真题 18-7：Memory Token 消耗控制
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain
# run: python 35_memory_token_control.py
# expected_runtime: <1s
# expected_output: memory config dump
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.7 (面试真题精选)
# Interview hooks:
#   1. 在生产环境中如何选择合适的 Memory 策略？需要考虑哪些权衡？
#   2. 为什么推荐用便宜的模型（如 gpt-4o-mini）做摘要？
# 推荐配置：Summary + Buffer 混合策略
class _MockLLM:
    model = "gpt-4o-mini"
    def invoke(self, msgs):
        class _R: content = "（mock）摘要"
        return _R()

memory_config = {
    "type": "ConversationSummaryBufferMemory",
    "llm": _MockLLM(),  # 摘要用便宜模型
    "max_token_limit": 2000,    # 总预算
    "return_messages": True,
}

print("=== Memory 配置 ===")
for k, v in memory_config.items():
    if k == "llm":
        print(f"  {k}: <{v.model}>")
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
