# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.5.1 itertools —— 迭代工具集
# difficulty: ⭐⭐⭐
# tier: core
# deps: itertools
# run: python 13_itertools.py
# expected_runtime: <1s
# expected_output: 演示 count/cycle/islice/chain/groupby/product 等
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.5.1 itertools
# Interview hooks:
#   1. itertools.islice 与 list[10:20] 的本质区别？
#   2. itertools.groupby 为什么需要先排序？
#   3. itertools.chain 与 chain.from_iterable 的差异？

"""
itertools —— Python 的迭代工具箱
面试常考：islice, chain, groupby, cycle
"""

import itertools

# ─────────────────────────────────────────────────────────────
# 无限迭代器
# ─────────────────────────────────────────────────────────────

# count(start, step) —— 无限计数
counter = itertools.count(10, 2)   # 10, 12, 14, 16, ...
print([next(counter) for _ in range(5)])   # [10, 12, 14, 16, 18]

# cycle(iterable) —— 无限循环
c = itertools.cycle("AB")
print([next(c) for _ in range(5)])   # ['A', 'B', 'A', 'B', 'A']

# repeat(elem, [n]) —— 重复元素
print(list(itertools.repeat("x", 3)))   # ['x', 'x', 'x']

# ─────────────────────────────────────────────────────────────
# 有限迭代器（面试高频）
# ─────────────────────────────────────────────────────────────

# islice —— 切片迭代器（不需要序列支持索引）
data = iter(range(100))
slice_10_20 = itertools.islice(data, 10, 20)   # 取第 10-19 个
print(list(slice_10_20))   # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# chain —— 连接多个迭代器
list1 = [1, 2, 3]
list2 = ["a", "b", "c"]
merged = itertools.chain(list1, list2)
print(list(merged))   # [1, 2, 3, 'a', 'b', 'c']

# 展平嵌套列表
nested = [[1, 2], [3, 4], [5, 6]]
flat = itertools.chain.from_iterable(nested)
print(list(flat))     # [1, 2, 3, 4, 5, 6]

# groupby —— 按连续相同值分组（面试常考）
data = ["A", "A", "B", "B", "B", "A", "C", "C"]
for key, group in itertools.groupby(data):
    print(f"{key}: {list(group)}")
# A: ['A', 'A']
# B: ['B', 'B', 'B']
# A: ['A']
# C: ['C', 'C']

# 注意：groupby 只对连续相同值分组！需要先排序
# 按首字母分组（需要先排序）
words = ["apple", "apricot", "banana", "blueberry", "cherry"]
words.sort()
for letter, group in itertools.groupby(words, key=lambda x: x[0]):
    print(f"{letter}: {list(group)}")
# a: ['apple', 'apricot']
# b: ['banana', 'blueberry']
# c: ['cherry']

# ─────────────────────────────────────────────────────────────
# 组合迭代器
# ─────────────────────────────────────────────────────────────

# product —— 笛卡尔积
print(list(itertools.product("AB", "12")))
# [('A', '1'), ('A', '2'), ('B', '1'), ('B', '2')]

# permutations —— 排列（有序，不重复）
print(list(itertools.permutations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# combinations —— 组合（无序，不重复）
print(list(itertools.combinations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]

# combinations_with_replacement —— 组合（允许重复）
print(list(itertools.combinations_with_replacement("ABC", 2)))
# [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]


if __name__ == "__main__":
    print("OK")
