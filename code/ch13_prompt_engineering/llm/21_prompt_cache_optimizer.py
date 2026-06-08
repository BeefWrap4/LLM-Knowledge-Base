# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.5 自适应缓存前缀管理器
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 21_prompt_cache_optimizer.py
# expected_runtime: <1s
# expected_output: 打印优化后的请求结构（cached_prefix + dynamic_part）
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.5
# Interview hooks:
# - 为何"稳定内容前置，动态内容后置"能提升命中率？
# - 命中率公式：cache_read / (cache_read + new_input)
# - 当 system_prompt 太短时该如何处理？(预热 / pad)


class PromptCacheOptimizer:
    """
    通过分析请求模式，自动优化 Prompt 缓存命中率
    核心思想：将"稳定不变的内容"前置，"动态变化的内容"后置
    """

    def __init__(self, min_cache_tokens: int = 1024):
        self.min_cache_tokens = min_cache_tokens
        self.prefix_hash_history = []  # 记录历史前缀哈希

    def split_prefix_suffix(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        将 messages 拆分为可缓存前缀 + 动态后缀
        拆分原则：
        1. system message 全部可缓存
        2. 早期 user/assistant 消息中内容稳定的部分可缓存
        3. 最后的 user 消息（当前问题）作为动态后缀
        """
        cacheable, dynamic = [], []
        for msg in messages:
            if msg["role"] == "system" or self._is_cacheable(msg):
                cacheable.append(msg)
            else:
                dynamic.append(msg)

        prefix_tokens = self._estimate_tokens(cacheable)
        if prefix_tokens < self.min_cache_tokens:
            # 前缀太短，全部作为 dynamic 走非缓存路径
            return [], messages
        return cacheable, dynamic

    def _is_cacheable(self, msg: dict) -> bool:
        content = msg.get("content", "")
        if not content:
            return False
        # 长内容通常可缓存（生产中应基于历史频率判断）
        return len(content) > 200

    def _estimate_tokens(self, messages: list[dict]) -> int:
        # 4 字符/token 启发式
        return sum(len(str(m)) for m in messages) // 4

    def build_request_with_cache(
        self, system_prompt: str, examples: list[dict], user_query: str, dynamic_context: str = ""
    ) -> dict:
        """
        构建最优缓存请求：
        1. system_prompt + examples 合并为强缓存前缀
        2. 动态文档 + user_query 作为变量
        """
        cached_prefix = {
            "type": "text",
            "text": system_prompt + "\n\n" + self._format_examples(examples),
        }
        dynamic_part = {
            "type": "text",
            "text": f"<context>{dynamic_context}</context>\n<query>{user_query}</query>",
        }
        return {
            "system": [cached_prefix],
            "messages": [{"role": "user", "content": [dynamic_part]}],
        }

    def _format_examples(self, examples: list[dict]) -> str:
        return "\n".join(
            f"示例{i + 1}：\n输入：{ex['input']}\n输出：{ex['output']}" for i, ex in enumerate(examples)
        )


if __name__ == "__main__":
    # 使用示例
    optimizer = PromptCacheOptimizer()
    request = optimizer.build_request_with_cache(
        system_prompt="你是一个 SQL 专家。" * 50,  # 重复以达到 1024+ tokens
        examples=[{"input": "...", "output": "..."}] * 5,
        user_query="查询最近 7 天的订单",
        dynamic_context="表结构：orders(id, user_id, amount, created_at)",
    )
    # 该请求可获得约 80-90% 的缓存命中率

    cached_text_len = len(request["system"][0]["text"])
    dynamic_text_len = len(request["messages"][0]["content"][0]["text"])
    print(f"[Cached Prefix] {cached_text_len} 字符 (~{cached_text_len // 4} tokens)")
    print(f"[Dynamic Part] {dynamic_text_len} 字符 (~{dynamic_text_len // 4} tokens)")
    print(f"[预期命中率] ~{cached_text_len / (cached_text_len + dynamic_text_len):.0%}")
