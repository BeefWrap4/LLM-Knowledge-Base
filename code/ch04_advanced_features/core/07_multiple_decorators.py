# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.2.4 多重装饰器
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: functools
# run: python 07_multiple_decorators.py
# expected_runtime: <1s
# expected_output: A-前 → B-前 → 目标 → B-后 → A-后
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.2.4 多重装饰器
# Interview hooks:
#   1. 多重装饰器的执行顺序是什么？（自上而下还是自下而上）
#   2. 装饰器嵌套时如何理解"洋葱模型"？
#   3. func = decorator_a(decorator_b(func)) 的执行顺序是？

"""
多重装饰器 —— 执行顺序是重点

@decorator_a
@decorator_b
def func():
    pass

等价于：func = decorator_a(decorator_b(func))
执行顺序：从内到外（先 decorator_b，再 decorator_a）
"""

from functools import wraps


def decorator_a(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("A - 前")
        result = func(*args, **kwargs)
        print("A - 后")
        return result

    return wrapper


def decorator_b(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("  B - 前")
        result = func(*args, **kwargs)
        print("  B - 后")
        return result

    return wrapper


@decorator_a
@decorator_b
def target():
    print("    目标函数")


target()
# A - 前
#   B - 前
#     目标函数
#   B - 后
# A - 后


if __name__ == "__main__":
    print("OK")
