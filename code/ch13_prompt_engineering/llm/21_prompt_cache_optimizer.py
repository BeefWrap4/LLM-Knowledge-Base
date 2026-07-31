# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.5 自适应缓存前缀管理器
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 21_prompt_cache_optimizer.py
# expected_runtime: <1s
# expected_output: 验证消息顺序不变并打印连续前缀/后缀
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.5
# Interview hooks:
# - 为何"稳定内容前置，动态内容后置"能提升命中率？
# - 命中率公式：cache_read / (cache_read + new_input)
# - 为什么不能用无意义 padding 凑缓存阈值？

from collections.abc import Callable


class PromptCachePlanner:
    """只在显式边界处分割连续前缀，不移动或改写任何消息。"""

    def __init__(
        self,
        count_tokens: Callable[[list[dict]], int],
        min_cache_tokens: int,
    ):
        self.count_tokens = count_tokens
        self.min_cache_tokens = min_cache_tokens

    def split(
        self,
        messages: list[dict],
        stable_prefix_count: int,
    ) -> tuple[list[dict], list[dict]]:
        if not 0 <= stable_prefix_count <= len(messages):
            raise ValueError("stable_prefix_count 越界")
        prefix = messages[:stable_prefix_count]
        suffix = messages[stable_prefix_count:]
        if self.count_tokens(prefix) < self.min_cache_tokens:
            return [], messages
        return prefix, suffix


if __name__ == "__main__":
    # 这里只用“词数”构造可运行教学测试；生产必须换成目标模型 tokenizer。
    def demo_token_counter(items: list[dict]) -> int:
        return sum(len(str(item.get("content", "")).split()) for item in items)

    messages = [
        {"role": "system", "content": "stable rules " * 20},
        {"role": "user", "content": "stable example input " * 10},
        {"role": "assistant", "content": "stable example output " * 10},
        {"role": "user", "content": "dynamic question"},
    ]
    planner = PromptCachePlanner(demo_token_counter, min_cache_tokens=40)
    prefix, suffix = planner.split(messages, stable_prefix_count=3)
    assert prefix + suffix == messages
    assert [m["role"] for m in prefix + suffix] == [m["role"] for m in messages]
    print(f"[连续稳定前缀] {len(prefix)} messages")
    print(f"[动态后缀] {len(suffix)} messages")
    print("OK")
