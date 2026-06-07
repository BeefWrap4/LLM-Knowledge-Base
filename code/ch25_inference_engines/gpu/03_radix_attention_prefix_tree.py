# ---
# chapter: 25
# topic: RadixAttention (SGLang)
# section: 25.2.2
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 03_radix_attention_prefix_tree.py
# expected_runtime: <1s
# expected_output: 演示 radix tree 复用共享前缀，统计 cache hit token 数
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.2
# Interview hooks:
#   1. RadixAttention 和 PagedAttention 的核心区别是什么？
#   2. system prompt / few-shot 场景为什么 RadixAttention 收益最大？
#   3. 什么时候 LRU evict radix 节点？(答: KV pool 满时从叶子开始)

"""RadixAttention: SGLang-style prefix tree for KV cache reuse.

Each radix node represents a span of tokens shared across requests.
When a new request's prompt matches an existing path, the matched
prefix reuses the cached KV blocks and the engine only computes the
suffix.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Iterable


class RadixNode:
    __slots__ = ("key", "children", "hit_count", "kv_block_ids")

    def __init__(self, key: tuple[int, ...] = ()) -> None:
        self.key: tuple[int, ...] = key
        self.children: dict[tuple[int, ...], "RadixNode"] = {}
        self.hit_count: int = 0
        # In real SGLang: list[int] of physical block ids holding the KV
        self.kv_block_ids: list[int] = []


class RadixCache:
    def __init__(self) -> None:
        self.root = RadixNode()

    def _find_longest_prefix(self, tokens: tuple[int, ...]) -> tuple[RadixNode, int]:
        """Return (deepest matching node, length of matched prefix)."""
        node = self.root
        matched = 0
        cur = tokens
        while cur:
            # Try to find a child whose key is a prefix of `cur`
            progressed = False
            for k, child in node.children.items():
                if cur[: len(k)] == k:
                    node = child
                    matched += len(k)
                    cur = cur[len(k):]
                    node.hit_count += 1
                    progressed = True
                    break
            if not progressed:
                break
        return node, matched

    def insert(self, tokens: Iterable[int], kv_block_ids: list[int]) -> int:
        toks = tuple(tokens)
        node, matched = self._find_longest_prefix(toks)
        if matched == len(toks):
            # Full hit; just attach the new kv to the existing node
            node.kv_block_ids = kv_block_ids
            return matched

        # Insert the remaining tail as a new child
        tail = toks[matched:]
        new_child = RadixNode(key=tail)
        new_child.kv_block_ids = kv_block_ids
        new_child.hit_count = 1
        node.children[tail] = new_child
        return matched

    def stats(self) -> dict:
        nodes = [self.root]

        def walk(n: RadixNode) -> None:
            for c in n.children.values():
                nodes.append(c)
                walk(c)
        walk(self.root)
        total_hits = sum(n.hit_count for n in nodes)
        return {"nodes": len(nodes), "total_hits": total_hits}


def main() -> None:
    cache = RadixCache()

    # Shared system prompt + few-shot examples
    system = (1, 2, 3, 4, 5, 6, 7, 8)  # "You are a helpful translator.\n\n"
    few_shot = (9, 10, 11, 12, 13, 14)  # Q: ... A: ...

    # 3 requests share system + few-shot, then diverge
    req_a_tail = (100, 101, 102)
    req_b_tail = (200, 201, 202, 203)
    req_c_tail = (100, 101, 102, 103)  # shares with A's tail

    for rid, tail in [(1, req_a_tail), (2, req_b_tail), (3, req_c_tail)]:
        prompt = system + few_shot + tail
        matched = cache.insert(prompt, kv_block_ids=[rid * 10, rid * 10 + 1])
        print(f"req {rid}: prompt_len={len(prompt)}  reused={matched}  "
              f"computed={len(prompt) - matched} tokens")

    # Fourth request: identical to A; should fully hit
    matched = cache.insert(system + few_shot + req_a_tail, kv_block_ids=[40, 41])
    print(f"req 4 (==A): reused={matched}  (full hit = 0 compute)")
    print("tree stats:", cache.stats())


if __name__ == "__main__":
    main()
