# ---
# chapter: 1
# topic: 集合 Set — 哈希集合
# section: 1.3.3
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 13_set_operations.py
# expected_runtime: <1s
# expected_output: 集合操作示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 734-785)
# Interview hooks:
#   1. set 和 dict 底层实现的相同点?
#   2. set 的 in 操作为什么是 O(1)?
#   3. frozenset 与 set 的区别及使用场景?
"""
集合：无序不重复元素集
底层实现：与字典相同的哈希表，只存键不存值
"""

# ─────────────────────────────────────────────────────────────
# 集合操作与复杂度
# ─────────────────────────────────────────────────────────────
# 操作              复杂度     说明
# ──────────────────────────────────
# add(x)            O(1)      添加元素
# remove(x)         O(1)      删除，不存在则 KeyError
# discard(x)        O(1)      删除，不存在不报错
# x in s            O(1)      成员判断（比 list 的 O(n) 快）
# s | t             O(len(s)+len(t))  并集
# s & t             O(min(len(s), len(t)))  交集
# s - t             O(len(s)) 差集

# ─────────────────────────────────────────────────────────────
# 集合的典型应用场景
# ─────────────────────────────────────────────────────────────


def find_duplicates(data: list) -> set:
    """利用集合快速查找重复元素"""
    seen = set()
    duplicates = set()
    for item in data:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates


def find_common(list1: list, list2: list) -> set:
    """查找两个列表的共同元素"""
    # 方式1：集合交集（O(n + m)）
    return set(list1) & set(list2)


# 演示
print(f"重复元素: {find_duplicates([1, 2, 3, 2, 4, 3, 5])}")
print(f"共同元素: {find_common([1, 2, 3, 4], [3, 4, 5, 6])}")

# ─────────────────────────────────────────────────────────────
# frozenset — 不可变集合（可作为字典键）
# ─────────────────────────────────────────────────────────────

fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError: 'frozenset' object has no attribute 'add'
print(f"frozenset: {fs}")

# frozenset 可作为字典键
cache = {
    frozenset(["a", "b"]): "组合ab",
    frozenset(["b", "c"]): "组合bc",
}
print(f"frozenset 作键: {cache[frozenset(['a', 'b'])]}")

if __name__ == "__main__":
    print("OK")
