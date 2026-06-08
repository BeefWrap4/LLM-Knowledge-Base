# ---
# chapter: 1
# topic: Lambda 与高阶函数
# section: 1.4.3
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 18_lambda_higher_order.py
# expected_runtime: <1s
# expected_output: Lambda/高阶函数示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 1041-1118)
# Interview hooks:
#   1. map/filter/reduce 与列表推导式的取舍?
#   2. functools.partial 与柯里化的关系?
#   3. lru_cache 如何将斐波那契从 O(2^n) 降到 O(n)?
"""
Lambda 与高阶函数 — 函数式编程基础
"""

import math
from functools import lru_cache, partial, reduce

# ─────────────────────────────────────────────────────────────
# Lambda 表达式
# ─────────────────────────────────────────────────────────────

# Lambda 是匿名函数，语法限制：只能有一个表达式，不能包含语句
square = lambda x: x**2
print(f"square(5) = {square(5)}")  # 25

# Lambda 的典型应用场景：作为回调函数
pairs = [(1, "one"), (2, "two"), (3, "three"), (4, "four")]
pairs.sort(key=lambda pair: pair[1])  # 按字符串排序
print(f"按字符串排序: {pairs}")  # [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]

# ─────────────────────────────────────────────────────────────
# 三大高阶函数：map / filter / reduce
# ─────────────────────────────────────────────────────────────

numbers = [1, 2, 3, 4, 5]

# map — 映射：对每个元素应用函数
squares = list(map(lambda x: x**2, numbers))
# 等价于 [x**2 for x in numbers]（列表推导式通常更推荐）
print(f"squares: {squares}")

# filter — 过滤：保留满足条件的元素
evens = list(filter(lambda x: x % 2 == 0, numbers))
# 等价于 [x for x in numbers if x % 2 == 0]
print(f"evens: {evens}")

# reduce — 累积：两两合并
product = reduce(lambda x, y: x * y, numbers)  # 1*2*3*4*5 = 120
print(f"product: {product}")

# ─────────────────────────────────────────────────────────────
# sorted 与自定义排序（面试高频）
# ─────────────────────────────────────────────────────────────

students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
    {"name": "David", "score": 92},
]

# 按分数降序，分数相同按姓名升序
sorted_students = sorted(students, key=lambda s: (-s["score"], s["name"]))
print(f"排序结果: {sorted_students}")
# [{'name': 'Bob', 'score': 92}, {'name': 'David', 'score': 92},
#  {'name': 'Alice', 'score': 85}, {'name': 'Charlie', 'score': 78}]

# ─────────────────────────────────────────────────────────────
# functools 工具函数
# ─────────────────────────────────────────────────────────────

# partial — 函数柯里化（固定部分参数）
base_2_log = partial(lambda base, x: math.log(x, base), 2)
print(f"log2(8) = {base_2_log(8)}")  # 3.0 (log2(8))


# lru_cache — 函数结果缓存（面试常考，用于记忆化递归）
@lru_cache(maxsize=128)
def fibonacci(n):
    """带缓存的斐波那契，时间复杂度从 O(2^n) 降到 O(n)"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print(f"fib(100) = {fibonacci(100)}")  # 354224848179261915075（瞬间完成）
print(f"缓存信息: {fibonacci.cache_info()}")

if __name__ == "__main__":
    print("OK")
