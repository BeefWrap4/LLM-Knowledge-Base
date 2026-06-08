# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.2.1 装饰器原理
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: functools
# run: python 04_decorator_basics.py
# expected_runtime: <1s
# expected_output: 装饰器前置/后置输出 + @wraps 元信息保留
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.2.1 装饰器原理
# Interview hooks:
#   1. @decorator 语法糖等价于什么？执行时机是定义时还是调用时？
#   2. 为什么需要 @functools.wraps？它解决了什么问题？
#   3. 装饰器是高阶函数吗？高阶函数的定义是什么？

"""
装饰器（Decorator）— 面试超高频考点

本质：装饰器是一个接收函数作为参数并返回函数的高阶函数
语法糖：@decorator 等价于 func = decorator(func)
"""

from functools import wraps

# ─────────────────────────────────────────────────────────────
# 最简单的装饰器
# ─────────────────────────────────────────────────────────────


def my_decorator(func):
    """装饰器函数 —— 接收一个函数，返回一个新函数"""

    @wraps(func)  # 保留原函数的元信息（__name__, __doc__ 等）
    def wrapper(*args, **kwargs):
        """包装函数 —— 在目标函数前后添加逻辑"""
        print(f"=== 调用 {func.__name__} 之前 ===")
        result = func(*args, **kwargs)  # 调用被装饰的函数
        print(f"=== 调用 {func.__name__} 之后 ===")
        return result

    return wrapper  # 返回包装函数


@my_decorator
def say_hello(name):
    """打招呼"""
    return f"Hello, {name}!"


# 等价于：say_hello = my_decorator(say_hello)

print(say_hello("Alice"))
# === 调用 say_hello 之前 ===
# === 调用 say_hello 之后 ===
# Hello, Alice!

# ─────────────────────────────────────────────────────────────
# 为什么需要 @wraps？
# ─────────────────────────────────────────────────────────────


def bad_decorator(func):
    """❌ 没有 wraps —— 丢失原函数元信息"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def good_decorator(func):
    """✅ 使用 wraps —— 保留原函数元信息"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@bad_decorator
def target():
    """目标函数"""
    pass


print(target.__name__)  # "wrapper" — 原函数名丢失！
print(target.__doc__)  # None


@good_decorator
def target2():
    """目标函数"""
    pass


print(target2.__name__)  # "target2" — 正确！
print(target2.__doc__)  # "目标函数"


if __name__ == "__main__":
    print("OK")
