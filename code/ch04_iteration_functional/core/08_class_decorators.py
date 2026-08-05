# ---
# chapter: 3
# topic: Python 函数、作用域与装饰器
# topic_id: iteration_functional.class_decorators
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: functools
# run: python 08_class_decorators.py
# expected_runtime: <1s
# expected_output: 两次调用计数 + 单例 Database is 检查
# ---
# See: ../../../03_Python函数作用域与装饰器.md
# Interview hooks:
#   1. 类作为装饰器时，__init__ 和 __call__ 各自承担什么角色？
#   2. 实现单例模式的装饰器思路是什么？有什么局限（线程安全）？
#   3. wraps(func)(self) 这种写法在做什么？

"""
类装饰器 —— 用类来实现装饰器

两种方式：
1. 类作为装饰器（实现 __call__）
2. 装饰器返回类
"""

from functools import wraps

# ─────────────────────────────────────────────────────────────
# 类作为装饰器（通过 __call__）
# ─────────────────────────────────────────────────────────────


class CountCalls:
    """类装饰器 —— 统计函数被调用次数"""

    def __init__(self, func):
        wraps(func)(self)  # 等价于 @wraps(func)，但 self 是类实例
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.__wrapped__.__name__} 被调用第 {self.count} 次")
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, owner):
        """支持实例方法绑定 —— 将实例绑定到第一个参数"""
        from functools import partial

        return partial(self.__call__, instance)


@CountCalls
def greet(name):
    return f"Hello {name}"


greet("Alice")  # 被调用第 1 次
greet("Bob")  # 被调用第 2 次
print(f"总调用次数: {greet.count}")

# ─────────────────────────────────────────────────────────────
# 类装饰器 —— 给类添加功能
# ─────────────────────────────────────────────────────────────


def singleton_class(cls):
    """类装饰器 —— 将任意类变为单例"""
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton_class
class Database:
    def __init__(self, url):
        self.url = url
        print(f"初始化数据库: {url}")


db1 = Database("mysql://localhost")
db2 = Database("postgresql://remote")
print(f"同一实例? {db1 is db2}")  # True
print(f"URL: {db2.url}")  # mysql://localhost


if __name__ == "__main__":
    print("OK")
