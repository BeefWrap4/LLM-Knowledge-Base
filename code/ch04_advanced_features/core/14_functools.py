# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.5.2 functools —— 函数式工具
# difficulty: ⭐⭐⭐
# tier: core
# deps: functools, operator, dataclasses
# run: python 14_functools.py
# expected_runtime: <1s
# expected_output: lru_cache 缓存命中 + reduce 累加 + partial 排序
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.5.2 functools
# Interview hooks:
#   1. lru_cache 内部数据结构是什么？复杂度多少？
#   2. functools.reduce 与 sum() 什么时候互换？什么时候不能？
#   3. partial 的应用场景是什么？与 lambda 相比的优缺点？

"""
functools —— 函数式编程工具
"""

from functools import lru_cache, partial, reduce, wraps

# ─────────────────────────────────────────────────────────────
# lru_cache —— 函数结果缓存（面试高频）
# ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def fibonacci(n):
    """带缓存的斐波那契 —— 时间复杂度 O(n)，原为 O(2^n)"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(100))  # 354224848179261915075（瞬间完成）
print(fibonacci.cache_info())  # CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# 缓存失效
fibonacci.cache_clear()

# ─────────────────────────────────────────────────────────────
# reduce —— 累积操作
# ─────────────────────────────────────────────────────────────

from operator import add, mul

numbers = [1, 2, 3, 4, 5]

# 累加
print(reduce(add, numbers))  # 15 — 等价于 sum()
# 累乘
print(reduce(mul, numbers))  # 120 — 1*2*3*4*5
# 带初始值
print(reduce(add, numbers, 100))  # 115 — 100+1+2+3+4+5

# 找最大值（自定义）
print(reduce(lambda x, y: x if x > y else y, numbers))  # 5

# ─────────────────────────────────────────────────────────────
# partial —— 函数部分应用
# ─────────────────────────────────────────────────────────────

from operator import mul

triple = partial(mul, 3)  # triple(x) == mul(3, x) == 3 * x
print(triple(5))  # 15

# 固定排序键
from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int


people = [Person("Alice", 30), Person("Bob", 25), Person("Charlie", 35)]
sort_by_age = partial(sorted, key=lambda p: p.age)
print(sort_by_age(people))

# ─────────────────────────────────────────────────────────────
# @wraps —— 保留原函数元信息（装饰器必备）
# ─────────────────────────────────────────────────────────────


def bare_decorator(func):
    """❌ 不使用 wraps"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def good_decorator(func):
    """✅ 使用 wraps"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@bare_decorator
def example1():
    """文档字符串"""
    pass


@good_decorator
def example2():
    """文档字符串"""
    pass


print(example1.__name__)  # "wrapper"
print(example1.__doc__)  # None
print(example2.__name__)  # "example2"
print(example2.__doc__)  # "文档字符串"


if __name__ == "__main__":
    print("OK")
